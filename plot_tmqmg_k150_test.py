"""
Plot pred vs reference parity charts for tmQMg k150 reproduce **test** predictions.

Style matches plot_vaska_barrier.py (5×5 scatter, diagonal, MAE/R² legend, metrics summary).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_PRED_DIR = "/data/jingyuan_data/tmqmg_reproduce/k150/predictions"
GLOB_PATTERN = "*_test_epoch*.json"

TARGET_LABELS = {
    "tzvp_dipole_moment": "dipole moment",
    "polarisability": "polarisability",
    "tzvp_homo_lumo_gap": "HOMO–LUMO gap",
}


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load truth/pred arrays and metadata from reproduce JSON."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    ys_true, ys_pred = [], []
    for rec in obj.get("predictions") or []:
        try:
            ys_true.append(float(rec["truth"]))
            ys_pred.append(float(rec["predicted"]))
        except (TypeError, ValueError, KeyError):
            continue

    meta = {k: obj[k] for k in ("graph", "target", "units", "split", "epoch", "n_molecules") if k in obj}
    return np.asarray(ys_true, dtype=float), np.asarray(ys_pred, dtype=float), meta


def r2_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    mean_y = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    mae = float(np.abs(y_true - y_pred).mean())
    return r2, mae


def _axis_limits(y_true: np.ndarray, y_pred: np.ndarray, pad_frac: float = 0.05) -> Tuple[float, float]:
    if y_true.size == 0:
        return 0.0, 1.0
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    pad = span * pad_frac
    return lo - pad, hi + pad


def _display_name(target: str) -> str:
    return TARGET_LABELS.get(target, target.replace("_", " "))


def plot_one(json_path: Path, out_png: Path) -> Tuple[float, float, int, dict]:
    """Returns (r2, mae, n, meta). Saves parity plot to out_png."""
    y_true, y_pred, meta = load_predictions(json_path)
    r2, mae = r2_mae(y_true, y_pred)
    n = int(y_true.size)

    target = str(meta.get("target", json_path.stem))
    units = str(meta.get("units", ""))
    label = _display_name(target)
    unit_suffix = f" ({units})" if units else ""

    vmin, vmax = _axis_limits(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    if n > 0:
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors="none")
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    ax.set_xlabel(f"Reference {label}{unit_suffix}")
    ax.set_ylabel(f"Predicted {label}{unit_suffix}")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    mae_label = f"MAE = {mae:.4f} {units}".strip() if units else f"MAE = {mae:.4f}"
    handles = [
        Line2D([0], [0], color="w", marker="", label=mae_label),
        Line2D([0], [0], color="w", marker="", label=f"R² = {r2:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, handlelength=0, handletextpad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r2, mae, n, meta


def _tag_from_stem(stem: str) -> Optional[str]:
    m = re.match(r"(.+)_test_epoch(\d+)$", stem)
    if m:
        return f"{m.group(1)}_epoch{m.group(2)}"
    m = re.match(r"(.+)_test$", stem)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot test-set parity charts for tmQMg k150 reproduce predictions."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_PRED_DIR,
        help=f"Folder with test prediction JSON files (default: {DEFAULT_PRED_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=GLOB_PATTERN,
        help=f"Glob under --dir (default: {GLOB_PATTERN})",
    )
    args = parser.parse_args()

    pred_dir = Path(args.dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {pred_dir}")

    paths = sorted(pred_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {pred_dir}")

    rows = []
    for p in paths:
        tag = _tag_from_stem(p.stem) or p.stem
        out_png = p.with_name(f"{p.stem}_plot.png")
        r2, mae, n, meta = plot_one(p, out_png)
        target = meta.get("target", tag)
        units = meta.get("units", "")
        rows.append((tag, target, units, str(p), mae, r2, n))
        print(f"[plot] {p.name} -> {out_png.name} | MAE={mae:.6f} {units}, R²={r2:.4f}, N={n}")

    summary_path = pred_dir / "tmqmg_k150_test_metrics_summary.txt"
    lines = [
        f"n_files = {len(rows)}",
        "",
        "per_file:",
    ]
    for tag, target, units, path_s, mae, r2, n in rows:
        lines.append(
            f"  {tag}  target={target}  units={units}  MAE={mae:.6f}  R2={r2:.6f}  N={n}  file={path_s}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
