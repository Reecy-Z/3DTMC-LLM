#!/usr/bin/env python3
"""
AutoGluon train/eval from descriptor CSV (e.g. DFT.csv / MAF.csv).

Main arguments:
  --feature-set   DFT | MAF (used for naming and logging)
  --desc-csv      Path to descriptor CSV
  --split         random | ligand
      random: 80/10/10 random split on mix_idx -> train / val / test
      ligand:   three ligand-scaffold OOD runs (Pybox / Biox / Biim as test)

"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore[import-untyped]

ROOT = os.path.dirname(os.path.abspath(__file__))
MIX_COL = "mix_idx"

META_COLS = {"mix_idx", "lmdb_key", "L_Scaffold"}
TARGET_CSV_COL = "ddG"
LABEL_COL = "label"
TEMP_COL = "Temp (K)"

ALL_LIGAND_SCAFFOLDS = {"Box", "Biox", "Pybox", "Pyox", "Biim", "Diamine"}

LIGAND_OOD_SPLITS = [
    {
        "name": "train_rest_test_Pybox",
        "test_types": {"Pybox"},
    },
    {
        "name": "train_rest_test_Biox",
        "test_types": {"Biox"},
    },
    {
        "name": "train_rest_test_Biim",
        "test_types": {"Biim"},
    },
]


def val_split_mix_ids(
    train_mix_ids: Sequence[int],
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Hold out validation mix_idx from training pool (sort, then shuffle with seed)."""
    rng = np.random.RandomState(seed)
    ids = sorted(int(i) for i in train_mix_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = ids[:n_val]
    tr_ids = ids[n_val:]
    return tr_ids, val_ids


def random_split_mix_ids(
    all_mix_ids: Sequence[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Random train / val / test split over all mix_idx."""
    rng = np.random.RandomState(seed)
    ids = sorted(int(i) for i in all_mix_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return train_ids, val_ids, test_ids


def row_idx_for_mix_ids(
    df: pd.DataFrame,
    mix_ids: Iterable[int],
    *,
    mix_col: str = "mix_idx",
) -> List[int]:
    """Map mix_idx list to DataFrame row indices (sorted by mix_idx ascending)."""
    wanted = set(int(i) for i in mix_ids)
    sub = df.loc[df[mix_col].isin(wanted)].sort_values(mix_col)
    return sub.index.tolist()


def _descriptor_sort_key(col: str) -> tuple[int, int | str]:
    try:
        return (0, int(col))
    except (ValueError, TypeError):
        return (1, str(col))


def _infer_descriptor_cols(df: pd.DataFrame) -> List[str]:
    skip = META_COLS | {TARGET_CSV_COL, LABEL_COL, TEMP_COL}
    desc_cols = [
        c
        for c in df.columns
        if c not in skip and np.issubdtype(df[c].dtype, np.number)
    ]
    desc_cols = sorted(desc_cols, key=_descriptor_sort_key)
    if TEMP_COL not in df.columns:
        raise SystemExit(f"CSV missing column: {TEMP_COL!r}")
    return [*desc_cols, TEMP_COL]


def load_descriptor_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    required = META_COLS | {TARGET_CSV_COL, TEMP_COL}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df[LABEL_COL] = pd.to_numeric(df[TARGET_CSV_COL], errors="coerce")
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    feature_cols = _infer_descriptor_cols(df)
    if not feature_cols:
        raise SystemExit("No numeric descriptor columns found")
    return df, feature_cols


def _build_tabular(df: pd.DataFrame, row_idx: List[int], feature_cols: List[str]) -> pd.DataFrame:
    part = df.loc[row_idx, feature_cols + [LABEL_COL]].copy()
    for c in feature_cols:
        part[c] = pd.to_numeric(part[c], errors="coerce").astype(np.float64)
    part[LABEL_COL] = pd.to_numeric(part[LABEL_COL], errors="coerce").astype(np.float64)
    return part.reset_index(drop=True)


def _fit_and_metrics(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    time_limit: int,
    presets: str,
) -> Dict[str, object]:
    from autogluon.tabular import TabularPredictor  # type: ignore[import-untyped]

    train_tab = _build_tabular(df, train_idx, feature_cols)
    val_tab = _build_tabular(df, val_idx, feature_cols) if val_idx else pd.DataFrame()
    test_tab = _build_tabular(df, test_idx, feature_cols)

    predictor = TabularPredictor(label=LABEL_COL, eval_metric="r2")
    predictor.fit(
        train_data=train_tab,
        tuning_data=val_tab if len(val_tab) else None,
        time_limit=time_limit,
        presets=presets,
    )

    def _metrics(name: str, tab: pd.DataFrame) -> Dict[str, float]:
        if tab.empty:
            return {"split": name, "n": 0, "MAE": float("nan"), "R2": float("nan")}
        y = tab[LABEL_COL].to_numpy(dtype=float)
        y_pred = np.asarray(predictor.predict(tab.drop(columns=[LABEL_COL])), dtype=float)
        return {
            "split": name,
            "n": len(tab),
            "MAE": float(mean_absolute_error(y, y_pred)),
            "R2": float(r2_score(y, y_pred)),
        }

    return {
        "train_n": len(train_idx),
        "val_n": len(val_idx),
        "test_n": len(test_idx),
        "train": _metrics("train", train_tab),
        "val": _metrics("val", val_tab),
        "test": _metrics("test", test_tab),
    }


def run_random(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
    time_limit: int,
    presets: str,
) -> Dict[str, object]:
    all_mix = df[MIX_COL].astype(int).tolist()
    tr_mix, va_mix, te_mix = random_split_mix_ids(
        all_mix, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    train_idx = row_idx_for_mix_ids(df, tr_mix, mix_col=MIX_COL)
    val_idx = row_idx_for_mix_ids(df, va_mix, mix_col=MIX_COL)
    test_idx = row_idx_for_mix_ids(df, te_mix, mix_col=MIX_COL)
    out = _fit_and_metrics(
        df, feature_cols, train_idx, val_idx, test_idx, time_limit, presets
    )
    out["experiment"] = "random_split"
    out["split_mode"] = "random"
    out["test_class"] = "random_holdout"
    out["seed"] = seed
    return out


def run_ligand_ood(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    experiments: List[Dict[str, object]] | None,
    val_frac: float,
    seed: int,
    time_limit: int,
    presets: str,
) -> List[Dict[str, object]]:
    pool = experiments or LIGAND_OOD_SPLITS
    results: List[Dict[str, object]] = []

    for spec in pool:
        test_types: Set[str] = spec["test_types"]  # type: ignore[assignment]
        train_mask = ~df["L_Scaffold"].isin(test_types)
        test_mask = df["L_Scaffold"].isin(test_types)

        train_mix = df.loc[train_mask, MIX_COL].astype(int).tolist()
        test_mix = df.loc[test_mask, MIX_COL].astype(int).tolist()
        tr_mix, val_mix = val_split_mix_ids(train_mix, val_frac=val_frac, seed=seed)
        tr_pos = row_idx_for_mix_ids(df, tr_mix, mix_col=MIX_COL)
        val_pos = row_idx_for_mix_ids(df, val_mix, mix_col=MIX_COL)
        test_pos = row_idx_for_mix_ids(df, test_mix, mix_col=MIX_COL)

        out = _fit_and_metrics(
            df, feature_cols, tr_pos, val_pos, test_pos, time_limit, presets
        )
        out.update(
            {
                "experiment": spec["name"],
                "split_mode": "ligand",
                "test_class": ",".join(sorted(test_types)),
                "train_n_total": len(train_mix),
                "test_n_ood": len(test_mix),
                "seed": seed,
            }
        )
        results.append(out)
        print(f"\n===== {spec['name']} =====")
        print(f"  test: {sorted(test_types)}")
        print(f"  train={out['train_n_total']}, test_ood={out['test_n_ood']}")
        for part in ("train", "val", "test"):
            m = out[part]
            print(f"  {m['split']}: n={m['n']}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")

    return results


def _flatten_results(
    results: List[Dict[str, object]],
    *,
    feature_set: str,
    desc_csv: str,
) -> pd.DataFrame:
    rows = []
    for out in results:
        rows.append(
            {
                "feature_set": feature_set,
                "desc_csv": desc_csv,
                "experiment": out.get("experiment"),
                "split_mode": out.get("split_mode"),
                "test_class": out.get("test_class"),
                "pool_n": out.get("pool_n", out.get("train_n", 0) + out.get("test_n", 0)),
                "train_n": out.get("train_n_total", out.get("train_n")),
                "val_n": out.get("val_n"),
                "test_n": out.get("test_n_ood", out.get("test_n")),
                "train_MAE": out["train"]["MAE"],
                "train_R2": out["train"]["R2"],
                "val_MAE": out["val"]["MAE"],
                "val_R2": out["val"]["R2"],
                "test_MAE": out["test"]["MAE"],
                "test_R2": out["test"]["R2"],
                "seed": out.get("seed"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon from descriptor CSV")
    parser.add_argument(
        "--feature-set",
        choices=("DFT", "MAF"),
        required=True,
        help="Descriptor type: DFT or MAF",
    )
    parser.add_argument(
        "--desc-csv",
        required=True,
        help="Descriptor CSV path, e.g. /path/to/NiComplex/DFT.csv or MAF.csv",
    )
    parser.add_argument(
        "--split",
        choices=("random", "ligand"),
        required=True,
        help="random=80/10/10 mix_idx split; ligand=Pybox/Biox/Biim OOD",
    )
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV (default: Data/autogluon_{DFT|MAF}_{random|ligand}.csv)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="In ligand mode, run only these experiment names",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.desc_csv):
        raise SystemExit(f"Descriptor CSV not found: {args.desc_csv}")

    df, feature_cols = load_descriptor_csv(args.desc_csv)
    print(f"feature_set: {args.feature_set}")
    print(f"desc_csv: {args.desc_csv}")
    print(f"n_samples: {len(df)}")
    print(f"n_features: {len(feature_cols)} (incl. {TEMP_COL})")
    print(f"split: {args.split}")
    print(f"seed: {args.seed} (mix_idx splits)")

    if args.split == "random":
        out = run_random(
            df,
            feature_cols,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            time_limit=args.time_limit,
            presets=args.presets,
        )
        out["pool_n"] = len(df)
        results = [out]
        print("\n===== random split =====")
        for part in ("train", "val", "test"):
            m = out[part]
            print(f"  {m['split']}: n={m['n']}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
    else:
        pool = LIGAND_OOD_SPLITS
        if args.experiments:
            names = set(args.experiments)
            pool = [s for s in LIGAND_OOD_SPLITS if s["name"] in names]
            if not pool:
                raise SystemExit(f"Unknown experiment(s): {args.experiments}")
        results = run_ligand_ood(
            df,
            feature_cols,
            experiments=pool,
            val_frac=args.val_frac,
            seed=args.seed,
            time_limit=args.time_limit,
            presets=args.presets,
        )
        for r in results:
            r["pool_n"] = len(df)

    out_path = args.out or os.path.join(
        ROOT,
        "Data",
        f"autogluon_{args.feature_set.lower()}_{args.split}.csv",
    )
    df_out = _flatten_results(results, feature_set=args.feature_set, desc_csv=args.desc_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
