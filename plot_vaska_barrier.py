"""
Plot pred vs ref for each pred_vaska_barrier_seed_*.jsonl under a results folder,
and report mean ± std of MAE and R² across seeds.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_RESULTS_DIR = "Vaska_Complex_Results"
GLOB_PATTERN = "pred_vaska_barrier_seed_*.json"


def load_predictions(path: Path):
    ys_true, ys_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return np.asarray(ys_true, dtype=float), np.asarray(ys_pred, dtype=float)

    # Supports both JSON list and JSONL formats.
    records = []
    if raw.startswith("["):
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            records = []
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for obj in records:
        try:
            y_true = float(obj["ref"])
            pv = obj.get("pred_value")
            if pv is None:
                continue
            y_pred = float(pv)
        except (TypeError, ValueError, KeyError):
            continue
        ys_true.append(y_true)
        ys_pred.append(y_pred)
    return np.asarray(ys_true, dtype=float), np.asarray(ys_pred, dtype=float)


def r2_mae(y_true: np.ndarray, y_pred: np.ndarray):
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    mean_y = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    mae = float(np.abs(y_true - y_pred).mean())
    return r2, mae


def _safe_std(values: list, ddof: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def plot_one(jsonl_path: Path, out_png: Path) -> tuple:
    """Returns (r2, mae, n). Saves parity plot to out_png."""
    y_true, y_pred = load_predictions(jsonl_path)
    r2, mae = r2_mae(y_true, y_pred)
    n = int(y_true.size)

    fig, ax = plt.subplots(figsize=(5, 5))
    if n > 0:
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors="none")
    ax.plot([0, 25], [0, 25], "k--", lw=1)
    ax.set_xlabel("DFT barrier (kcal/mol)")
    ax.set_ylabel("Predicted barrier (kcal/mol)")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_aspect("equal")
    handles = [
        Line2D([0], [0], color="w", marker="", label=f"MAE = {mae:.3f} kcal/mol"),
        Line2D([0], [0], color="w", marker="", label=f"R² = {r2:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, handlelength=0, handletextpad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r2, mae, n


def _seed_from_stem(stem: str) -> Optional[str]:
    m = re.match(r"pred_vaska_barrier_seed_(\d+)$", stem)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot parity charts for each seed JSONL and summarize MAE/R² across seeds."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Folder containing {GLOB_PATTERN} (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=GLOB_PATTERN,
        help="Glob pattern under --dir (default: pred_vaska_barrier_seed_*.jsonl)",
    )
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    paths = sorted(results_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {results_dir}")

    maes, r2s = [], []
    rows = []

    for p in paths:
        seed = _seed_from_stem(p.stem) or p.stem
        out_png = p.with_name(f"{p.stem}_plot.png")
        r2, mae, n = plot_one(p, out_png)
        maes.append(mae)
        r2s.append(r2)
        rows.append((seed, str(p), mae, r2, n))
        print(f"[plot] {p.name} -> {out_png.name} | MAE={mae:.4f}, R²={r2:.4f}, N={n}")

    mae_mean = float(np.nanmean(maes)) if maes else float("nan")
    mae_std = _safe_std(maes)
    r2_mean = float(np.nanmean(r2s)) if r2s else float("nan")
    r2_std = _safe_std(r2s)

    summary_path = results_dir / "vaska_barrier_metrics_summary.txt"
    lines = [
        f"n_seeds = {len(rows)}",
        f"MAE mean = {mae_mean:.6f} kcal/mol, std = {mae_std:.6f}",
        f"R²  mean = {r2_mean:.6f}, std = {r2_std:.6f}",
        "",
        "per_seed:",
    ]
    for seed, path_s, mae, r2, n in rows:
        lines.append(f"  seed={seed}  MAE={mae:.6f}  R2={r2:.6f}  N={n}  file={path_s}")

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Saved summary: {summary_path}")
    print(f"Across {len(rows)} seeds: MAE = {mae_mean:.4f} ± {mae_std:.4f} kcal/mol, R² = {r2_mean:.4f} ± {r2_std:.4f}")


if __name__ == "__main__":
    main()
