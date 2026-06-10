#!/usr/bin/env python3
"""
FAISS K-means clustering + cluster-based train/test split.

Requires the tmc-split conda env (see environment-split.yml):
  conda activate tmc-split

Test clusters: whole clusters farthest from the dataset body centroid (size-weighted
mean of cluster centroids), for train/test separation in fingerprint space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from complex_fingerprint import (
    ComplexFingerprint,
    build_complex_fingerprint,
    vector_for_clustering,
)

faiss = None
FAISS_AVAILABLE = False


def _ensure_faiss() -> None:
    """Import FAISS once; raise with install hint if unavailable."""
    global faiss, FAISS_AVAILABLE
    if FAISS_AVAILABLE:
        return
    try:
        import faiss as _faiss

        faiss = _faiss
        FAISS_AVAILABLE = True
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "FAISS not available (often NumPy 2 incompatibility). "
            "Use: conda activate tmc-split  "
            "(or: conda install -c conda-forge 'numpy<2' faiss-cpu)"
        ) from exc


def normalize_l2_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-8)


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str,
    *,
    w_bits: float,
    w_meta: float,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return (X, valid_df, errors). X is float32 and L2-normalized per row."""
    rows: list[np.ndarray] = []
    keep_idx: list[int] = []
    errors: list[str] = []
    metals: list[str] = []
    charges: list[int] = []
    d_elecs: list[int] = []

    for i, smi in enumerate(tqdm(df[smiles_col].astype(str), desc="Fingerprints")):
        fp = build_complex_fingerprint(smi)
        if fp is None:
            errors.append(f"row {i}: invalid SMILES or no transition metal")
            continue
        rows.append(vector_for_clustering(fp, w_bits=w_bits, w_meta=w_meta))
        keep_idx.append(i)
        metals.append(fp.metal_symbol)
        charges.append(fp.formal_charge)
        d_elecs.append(fp.d_electrons)

    if not rows:
        raise RuntimeError("No valid fingerprints built.")

    X = np.vstack(rows).astype(np.float32)
    X = normalize_l2_matrix(X)
    valid_df = df.iloc[keep_idx].reset_index(drop=True)
    valid_df["metal_symbol"] = metals
    valid_df["formal_charge"] = charges
    valid_df["d_electrons"] = d_elecs
    return X, valid_df, errors


def build_feature_matrix_from_npz(
    df: pd.DataFrame,
    id_col: str,
    npz_path: Path,
    *,
    w_bits: float,
    w_meta: float,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Load precomputed vectors; align rows to df via id_col (e.g. CSD code)."""
    data = np.load(npz_path, allow_pickle=True)
    csd_codes = data["csd_codes"]
    vectors = data["vectors"]
    metals = data["metal_symbol"]
    charges = data["formal_charge"]
    d_elecs = data["d_electrons"]
    csd_to_i = {str(c): i for i, c in enumerate(csd_codes)}

    rows: list[np.ndarray] = []
    keep_idx: list[int] = []
    errors: list[str] = []
    metals_out: list[str] = []
    charges_out: list[int] = []
    d_elecs_out: list[int] = []

    ids = df[id_col].astype(str).to_numpy()
    for i in tqdm(range(len(df)), desc="Load fingerprints"):
        csd = ids[i]
        j = csd_to_i.get(csd)
        if j is None:
            errors.append(f"row {i} ({csd}): no precomputed fingerprint")
            continue
        fp = ComplexFingerprint(
            vector=vectors[j].astype(np.float32, copy=False),
            metal_symbol=str(metals[j]),
            formal_charge=int(charges[j]),
            d_electrons=int(d_elecs[j]),
            smiles="",
        )
        rows.append(vector_for_clustering(fp, w_bits=w_bits, w_meta=w_meta))
        keep_idx.append(i)
        metals_out.append(fp.metal_symbol)
        charges_out.append(fp.formal_charge)
        d_elecs_out.append(fp.d_electrons)

    if not rows:
        raise RuntimeError(f"No rows matched fingerprints in {npz_path}")

    X = np.vstack(rows).astype(np.float32)
    X = normalize_l2_matrix(X)
    valid_df = df.iloc[keep_idx].reset_index(drop=True)
    valid_df["metal_symbol"] = metals_out
    valid_df["formal_charge"] = charges_out
    valid_df["d_electrons"] = d_elecs_out
    print(f"Loaded cache: {npz_path} ({len(csd_codes):,} entries)")
    return X, valid_df, errors


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    niter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels[N], centroids[K, d]) using spherical FAISS K-means."""
    _ensure_faiss()
    d = X.shape[1]
    kmeans = faiss.Kmeans(
        d,
        n_clusters,
        niter=niter,
        verbose=False,
        seed=seed,
        spherical=True,
    )
    kmeans.train(X)
    _, labels = kmeans.index.search(X, 1)
    labels = labels.reshape(-1).astype(np.int32)
    centroids = kmeans.centroids.copy().astype(np.float32)
    centroids = normalize_l2_matrix(centroids)
    return labels, centroids


def _cluster_sizes(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_clusters)
    return counts.astype(np.int64)


def select_test_clusters_far_from_train(
    centroids: np.ndarray,
    cluster_sizes: np.ndarray,
    n_samples: int,
    *,
    test_frac_min: float,
    test_frac_max: float,
) -> list[int]:
    """
    Pick whole clusters farthest from the dataset body centroid (size-weighted
    mean of cluster centroids). Add farthest-first until test_frac_min; drop
    the closest-among-selected if above test_frac_max (keeps cluster purity).
    """
    weights = cluster_sizes.astype(np.float64)
    body = (centroids * weights[:, None]).sum(axis=0) / max(weights.sum(), 1.0)
    body = normalize_l2_matrix(body.reshape(1, -1)).reshape(-1)
    dist = 1.0 - (centroids @ body)

    order = np.argsort(-dist)
    selected: list[int] = []
    count = 0
    min_target = int(np.ceil(test_frac_min * n_samples))
    max_target = int(np.floor(test_frac_max * n_samples))

    for j in order:
        j = int(j)
        selected.append(j)
        count += int(cluster_sizes[j])
        if count >= min_target:
            break

    while count > max_target and len(selected) > 1:
        rem = min(selected, key=lambda cid: dist[cid])
        selected.remove(rem)
        count -= int(cluster_sizes[rem])

    return selected


def assign_splits(
    labels: np.ndarray,
    test_clusters: list[int],
    n_samples: int,
    *,
    test_frac_max: float,
    target_test_frac: float | None,
    seed: int,
) -> np.ndarray:
    """
    Return split array: 0=train, 1=test.
    Subsample test if above test_frac_max or target_test_frac.
    """
    test_set = set(test_clusters)
    is_test = np.isin(labels, list(test_set))
    test_idx = np.flatnonzero(is_test)
    n_test = len(test_idx)

    cap = int(np.floor(test_frac_max * n_samples))
    if target_test_frac is not None:
        cap = min(cap, int(np.round(target_test_frac * n_samples)))

    if n_test > cap and cap > 0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(test_idx, size=cap, replace=False)
        is_test = np.zeros(n_samples, dtype=bool)
        is_test[keep] = True

    return is_test.astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAISS cluster train/test split via ComplexFingerprint (far-from-train)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/path/to/tmQMg/all.csv"),
    )
    parser.add_argument("--smiles-col", default="SMILES_CSD_fixed")
    parser.add_argument("--id-col", default="CSD_code")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/path/to/tmQM_faiss_cluster_split.csv"),
    )
    parser.add_argument("--n-clusters", type=int, default=150)
    parser.add_argument("--kmeans-niter", type=int, default=25)
    parser.add_argument("--test-frac-min", type=float, default=0.10)
    parser.add_argument("--test-frac-max", type=float, default=0.20)
    parser.add_argument(
        "--target-test-frac",
        type=float,
        default=None,
        help="If set, subsample test to this fraction when above cap (within max).",
    )
    parser.add_argument("--w-bits", type=float, default=0.7)
    parser.add_argument("--w-meta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug subsample.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Write train.csv and test.csv here (original input columns only).",
    )
    parser.add_argument(
        "--fingerprints-npz",
        type=Path,
        default=None,
        help="Precomputed .npz from build_tmqmg_fingerprints.py (skips RDKit).",
    )
    args = parser.parse_args()

    if not (0 < args.test_frac_min <= args.test_frac_max < 1):
        raise SystemExit("Require 0 < test_frac_min <= test_frac_max < 1")

    df = pd.read_csv(args.input)
    input_cols = list(df.columns)
    df = df.dropna(subset=[args.smiles_col])
    if args.max_samples:
        df = df.head(args.max_samples)

    if args.fingerprints_npz is not None:
        X, valid_df, errors = build_feature_matrix_from_npz(
            df,
            args.id_col,
            args.fingerprints_npz,
            w_bits=args.w_bits,
            w_meta=args.w_meta,
        )
    else:
        X, valid_df, errors = build_feature_matrix(
            df,
            args.smiles_col,
            w_bits=args.w_bits,
            w_meta=args.w_meta,
        )
    n = X.shape[0]
    print(f"Valid complexes: {n:,} / {len(df):,} ({len(errors):,} skipped)")

    labels, centroids = run_kmeans(
        X,
        args.n_clusters,
        niter=args.kmeans_niter,
        seed=args.seed,
    )
    sizes = _cluster_sizes(labels, args.n_clusters)
    test_clusters = select_test_clusters_far_from_train(
        centroids,
        sizes,
        n,
        test_frac_min=args.test_frac_min,
        test_frac_max=args.test_frac_max,
    )
    is_test = assign_splits(
        labels,
        test_clusters,
        n,
        test_frac_max=args.test_frac_max,
        target_test_frac=args.target_test_frac,
        seed=args.seed,
    )

    out = valid_df[
        [args.id_col, args.smiles_col, "metal_symbol", "formal_charge", "d_electrons"]
    ].copy()
    out["cluster_id"] = labels
    out["split"] = np.where(is_test, "test", "train")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    if args.export_dir is not None:
        split_col = np.where(is_test, "test", "train")
        export_base = valid_df[input_cols].copy()
        export_base["_split"] = split_col
        args.export_dir.mkdir(parents=True, exist_ok=True)
        export_base.loc[export_base["_split"] == "train"].drop(columns="_split").to_csv(
            args.export_dir / "train.csv", index=False
        )
        export_base.loc[export_base["_split"] == "test"].drop(columns="_split").to_csv(
            args.export_dir / "test.csv", index=False
        )
        print(f"Wrote {args.export_dir / 'train.csv'}")
        print(f"Wrote {args.export_dir / 'test.csv'}")

    meta_path = args.output.with_suffix(".json")
    n_test = int(is_test.sum())
    summary = {
        "n_valid": n,
        "n_skipped": len(errors),
        "fingerprints_npz": str(args.fingerprints_npz) if args.fingerprints_npz else None,
        "backend": "faiss",
        "n_clusters": args.n_clusters,
        "test_clusters": test_clusters,
        "n_test": n_test,
        "n_train": n - n_test,
        "test_frac": n_test / n,
        "test_frac_min": args.test_frac_min,
        "test_frac_max": args.test_frac_max,
        "selection_mode": "far_from_train",
        "w_bits": args.w_bits,
        "w_meta": args.w_meta,
        "seed": args.seed,
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Test clusters ({len(test_clusters)}): {test_clusters}")
    print(f"Train: {n - n_test:,} | Test: {n_test:,} ({100 * n_test / n:.2f}%)")
    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
