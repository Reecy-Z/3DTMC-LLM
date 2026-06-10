"""Unified Fig.8-style GP evaluation: random split or LOLO OOD.

Usage:
    python run_gp.py --data data/Vaska.csv --mode random
    python run_gp.py --data data/Vaska.csv --mode random --seed 3
    python run_gp.py --data data/Vaska.csv --mode OOD

- random: 80/10/10 train/val/test random split; --seed defaults to 1.
- OOD:    leave-one-ligand-out using ligand_a1/a2/c/b columns in Vaska.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, rdFingerprintGenerator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# --- GP pipeline constants ---
FA_PROPS = ("chi", "Z", "I", "T", "S")
FA3_DEPTH = 3
RDKIT_MAX_PATH = 5
MORGAN_RADIUS = 3
FP_SIZE = 16384
TOP_K = 300
GB_ESTIMATORS = 100
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

HALIDE_CODES = frozenset({"chloride", "fluoride", "bromide", "iodide"})
LIGAND_CODE_COLS = ("ligand_a1", "ligand_a2", "ligand_c", "ligand_b")
LIGAND_NAME_COLS = ("ligand_a1_name", "ligand_a2_name", "ligand_c_name", "ligand_b_name")
_MORGAN_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=FP_SIZE
)


def fa3_columns() -> List[str]:
    return [f"{prop}-{depth}_fa" for depth in range(FA3_DEPTH + 1) for prop in FA_PROPS]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smiles_to_fps(smiles: str) -> Tuple[np.ndarray, np.ndarray] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdkit_fp = np.zeros(FP_SIZE, dtype=np.float32)
    morgan_fp = np.zeros(FP_SIZE, dtype=np.float32)
    rdkit_bits = RDKFingerprint(mol, maxPath=RDKIT_MAX_PATH, fpSize=FP_SIZE)
    morgan_bits = _MORGAN_FP_GEN.GetFingerprint(mol)
    Chem.DataStructs.ConvertToNumpyArray(rdkit_bits, rdkit_fp)
    Chem.DataStructs.ConvertToNumpyArray(morgan_bits, morgan_fp)
    return rdkit_fp, morgan_fp


def build_feature_matrix(
    df: pd.DataFrame, cache_path: Path | None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"], data["valid_idx"].tolist()

    fa3 = df[fa3_columns()].to_numpy(dtype=np.float32)
    y = df["barrier"].to_numpy(dtype=np.float64)
    n = len(df)
    rdkit_mat = np.zeros((n, FP_SIZE), dtype=np.float32)
    morgan_mat = np.zeros((n, FP_SIZE), dtype=np.float32)
    valid_idx: List[int] = []

    for i, smi in enumerate(df["smiles"]):
        fps = smiles_to_fps(str(smi))
        if fps is None:
            continue
        rdkit_fp, morgan_fp = fps
        rdkit_mat[i] = rdkit_fp
        morgan_mat[i] = morgan_fp
        valid_idx.append(i)

    X = np.hstack([fa3, rdkit_mat, morgan_mat]).astype(np.float64)
    y = y[valid_idx]
    X = X[valid_idx]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, valid_idx=np.array(valid_idx, dtype=np.int32))

    return X, y, valid_idx


def split_train_val_test(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def select_top_features(X_train: np.ndarray, y_train: np.ndarray, seed: int, top_k: int) -> np.ndarray:
    gb = GradientBoostingRegressor(
        n_estimators=GB_ESTIMATORS,
        random_state=seed,
    )
    gb.fit(X_train, y_train)
    importances = gb.feature_importances_
    k = min(top_k, X_train.shape[1])
    return np.argsort(importances)[-k:]


def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=seed,
    )
    gp.fit(X_train_s, y_train)
    return gp, scaler


def predict_gp(gp: GaussianProcessRegressor, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return gp.predict(scaler.transform(X))


def fit_predict_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    gp, scaler = fit_gp(X_train, y_train, seed)
    return predict_gp(gp, scaler, X_test)


def evaluate_seed(
    X: np.ndarray, y: np.ndarray, seed: int, top_k: int
) -> Tuple[float, float, float, float, int, int, int]:
    train_idx, val_idx, test_idx = split_train_val_test(len(y), seed)
    X_train_full, y_train = X[train_idx], y[train_idx]
    X_val_full, y_val = X[val_idx], y[val_idx]
    X_test_full, y_test = X[test_idx], y[test_idx]

    feat_idx = select_top_features(X_train_full, y_train, seed, top_k)
    X_train = X_train_full[:, feat_idx]
    X_val = X_val_full[:, feat_idx]
    X_test = X_test_full[:, feat_idx]

    gp, scaler = fit_gp(X_train, y_train, seed)
    y_val_pred = predict_gp(gp, scaler, X_val)
    y_test_pred = predict_gp(gp, scaler, X_test)
    return (
        r2_score(y_val, y_val_pred),
        mae(y_val, y_val_pred),
        r2_score(y_test, y_test_pred),
        mae(y_test, y_test_pred),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )


def parse_ligand_tokens(filename: str) -> List[str]:
    parts = filename.split("_")
    ligs = [p for p in parts if p.startswith("dft-") or p in HALIDE_CODES]
    if len(ligs) != 4:
        raise ValueError(f"Expected 4 ligand tokens in {filename!r}, got {ligs}")
    return ligs


def _has_ligand_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in LIGAND_CODE_COLS)


def load_ligand_codes(df: pd.DataFrame) -> List[str]:
    if _has_ligand_columns(df):
        codes: set[str] = set()
        for col in LIGAND_CODE_COLS:
            codes.update(df[col].dropna().astype(str))
        return sorted(codes)
    raise ValueError(
        f"OOD mode requires ligand columns {LIGAND_CODE_COLS} in the data CSV. "
        "Use data/Vaska.csv or add these columns."
    )


def load_ligand_names(df: pd.DataFrame) -> Dict[str, str]:
    if not _has_ligand_columns(df):
        return {}

    mapping: Dict[str, str] = {}
    has_name_cols = all(col in df.columns for col in LIGAND_NAME_COLS)
    for row in df.itertuples(index=False):
        for code_col, name_col in zip(LIGAND_CODE_COLS, LIGAND_NAME_COLS):
            code = getattr(row, code_col, None)
            if code is None or pd.isna(code) or code in mapping:
                continue
            if has_name_cols:
                name = getattr(row, name_col, None)
                mapping[str(code)] = str(name) if name is not None and not pd.isna(name) else str(code)
            else:
                mapping[str(code)] = str(code)
    return mapping


def build_ligand_lists(df: pd.DataFrame, valid_idx: List[int]) -> List[List[str]]:
    if _has_ligand_columns(df):
        return [df.iloc[i][list(LIGAND_CODE_COLS)].astype(str).tolist() for i in valid_idx]
    return [parse_ligand_tokens(df.iloc[i]["filename"]) for i in valid_idx]


def lolo_masks(ligand_lists: List[List[str]], holdout: str) -> Tuple[np.ndarray, np.ndarray]:
    test_mask = np.array([holdout in ligs for ligs in ligand_lists], dtype=bool)
    train_mask = ~test_mask
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Empty split for holdout={holdout!r}")
    return train_mask, test_mask


def evaluate_ood_fold(
    X: np.ndarray,
    y: np.ndarray,
    ligand_lists: List[List[str]],
    holdout: str,
    ligand_name: str,
    seed: int,
    top_k: int,
) -> Dict[str, float | int | str]:
    train_mask, test_mask = lolo_masks(ligand_lists, holdout)
    X_train_full, y_train = X[train_mask], y[train_mask]
    X_test_full, y_test = X[test_mask], y[test_mask]

    feat_idx = select_top_features(X_train_full, y_train, seed, top_k)
    y_pred = fit_predict_gp(
        X_train_full[:, feat_idx],
        y_train,
        X_test_full[:, feat_idx],
        seed,
    )
    return {
        "holdout_ligand": holdout,
        "ligand_name": ligand_name,
        "n_train": int(train_mask.sum()),
        "n_ood_test": int(test_mask.sum()),
        "r2": r2_score(y_test, y_pred),
        "mae": mae(y_test, y_pred),
    }


def build_ood_summary(
    rows: List[Dict[str, float | int | str]],
    seed: int,
    top_k: int,
    holdout_plan: List[str],
    status: str,
) -> Dict:
    r2_vals = np.array([r["r2"] for r in rows], dtype=float)
    mae_vals = np.array([r["mae"] for r in rows], dtype=float)
    summary: Dict = {
        "protocol": "leave-one-ligand-out",
        "holdout_plan": holdout_plan,
        "seed": seed,
        "top_k": top_k,
        "status": status,
        "n_completed_folds": len(rows),
        "n_total_folds": len(holdout_plan),
        "folds": rows,
    }
    if rows:
        summary["macro_mean_r2"] = float(r2_vals.mean())
        summary["macro_mean_mae"] = float(mae_vals.mean())
    return summary


def _cache_path(data_path: Path) -> Path:
    return data_path.parent / f"{data_path.stem}_fig8_features.npz"


def _out_path(data_path: Path, mode: str, seed: int) -> Path:
    if mode == "random":
        return data_path.parent / f"{data_path.stem}_gp_random_seed{seed}.json"
    return data_path.parent / f"{data_path.stem}_gp_ood_lolo.json"


def run_random(data_path: Path, seed: int) -> dict:
    df = pd.read_csv(data_path)
    cache = _cache_path(data_path)
    print(f"Loading features from {cache if cache.exists() else 'building cache'}...")
    X, y, _ = build_feature_matrix(df, cache)
    print(f"Samples: {len(y)}, feature dim: {X.shape[1]}")

    val_r2, val_mae, test_r2, test_mae, n_train, n_val, n_test = evaluate_seed(X, y, seed, TOP_K)
    result = {
        "mode": "random",
        "data": str(data_path),
        "seed": seed,
        "split": "80/10/10",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "val_r2": val_r2,
        "val_mae": val_mae,
        "r2": test_r2,
        "mae": test_mae,
    }
    print(
        f"seed={seed}  train={n_train}  val={n_val}  test={n_test}  "
        f"val R2={val_r2:.4f} MAE={val_mae:.3f}  "
        f"test R2={test_r2:.4f} MAE={test_mae:.3f} kcal/mol"
    )
    return result


def run_ood(data_path: Path, seed: int) -> dict:
    df = pd.read_csv(data_path)
    cache = _cache_path(data_path)
    print(f"Loading features from {cache if cache.exists() else 'building cache'}...")
    X, y, valid_idx = build_feature_matrix(df, cache)
    ligand_lists = build_ligand_lists(df, valid_idx)

    holdout_plan = load_ligand_codes(df)
    code_to_name = load_ligand_names(df)
    print(f"Samples: {len(y)}, LOLO folds: {len(holdout_plan)}")

    rows = []
    for holdout in holdout_plan:
        name = code_to_name.get(holdout, holdout)
        print(f"\n=== LOLO holdout: {holdout} ({name}) ===")
        row = evaluate_ood_fold(X, y, ligand_lists, holdout, name, seed, TOP_K)
        rows.append(row)
        print(
            f"OOD test={row['n_ood_test']} train={row['n_train']}  "
            f"R2={row['r2']:.4f}  MAE={row['mae']:.3f} kcal/mol"
        )

    summary = build_ood_summary(rows, seed, TOP_K, holdout_plan, "completed")
    summary["mode"] = "OOD"
    summary["data"] = str(data_path)
    print("\n=== Macro average over LOLO folds ===")
    print(f"mean R2  = {summary['macro_mean_r2']:.4f}")
    print(f"mean MAE = {summary['macro_mean_mae']:.3f} kcal/mol")
    return summary


def _parse_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in {"random", "ood"}:
        raise argparse.ArgumentTypeError("mode must be 'random' or 'OOD'")
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig.8-style GP evaluation (random or LOLO OOD)")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to Vaska.csv (features + ligand columns)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=_parse_mode,
        help="Evaluation mode: random (80/10/10 split) or OOD (leave-one-ligand-out)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/test split (random mode) and GB/GP (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if args.mode == "random":
        result = run_random(data_path, args.seed)
    else:
        result = run_ood(data_path, args.seed)

    out_path = _out_path(data_path, args.mode, args.seed)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
