#!/usr/bin/env python3
"""UMAP visualization of tmQMg FAISS train/test split (ComplexFingerprint space)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(_ROOT / ".numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from tqdm import tqdm

from split_train_test_faiss import build_feature_matrix_from_npz

POINT_SIZE = 6

# train = background grey (tmQM style); test = highlighted light blue (OMol25 style)
SPLIT_COLORS = {
    "train": "#B0B0B0",
    "test": "#7EB6E6",
}
PANEL_LABEL = "a"
PANEL_HEADER_Y = 1.12
PANEL_LABEL_X = -0.10
LEGEND_MARKER_SIZE = 8.0
AXIS_LABEL_FONTSIZE = 7
LEGEND_FONTSIZE = 7

def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

def add_panel_header(ax: plt.Axes, panel_label: str, title: str | None = None) -> None:
    ax.text(
        PANEL_LABEL_X,
        PANEL_HEADER_Y,
        panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )
    if title:
        ax.text(
            0.5,
            PANEL_HEADER_Y,
            title,
            transform=ax.transAxes,
            fontsize=7,
            fontweight="normal",
            va="top",
            ha="center",
            clip_on=False,
        )


def load_split_metadata(json_path: Path | None) -> dict:
    if json_path is None or not json_path.is_file():
        return {"w_bits": 0.7, "w_meta": 0.3, "seed": 42, "n_clusters": 150}
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return meta


def n_clusters_from_meta(meta: dict) -> int:
    return int(meta.get("n_clusters", 150))


def panel_title_cluster(meta: dict) -> str:
    k = n_clusters_from_meta(meta)
    return f"ComplexFingerprint\n(FAISS K={k} split, L2 norm)"


def panel_title_raw(meta: dict) -> str:
    k = n_clusters_from_meta(meta)
    return f"ComplexFingerprint (raw)\n(FAISS K={k} split)"


def suptitle_for(meta: dict, *, raw: bool) -> str:
    k = n_clusters_from_meta(meta)
    if raw:
        return f"ComplexFingerprint (raw, K={k})"
    return f"ComplexFingerprint (FAISS K={k})"


def default_output_prefix(split_csv: Path) -> Path:
    return split_csv.with_name(f"{split_csv.stem}_umap")


def stratified_fit_indices(
    splits: np.ndarray,
    n_fit: int,
    seed: int,
) -> np.ndarray:
    """Sample ~half train / half test for UMAP fitting."""
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    n_train = min(len(train_idx), max(1, n_fit // 2))
    n_test = min(len(test_idx), max(1, n_fit - n_train))
    pick_train = rng.choice(train_idx, size=n_train, replace=False)
    pick_test = rng.choice(test_idx, size=n_test, replace=False)
    return np.concatenate([pick_train, pick_test])


def run_umap(
    X: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    fit_indices: np.ndarray | None,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_jobs=1,
        verbose=False,
    )
    if fit_indices is not None and len(fit_indices) < len(X):
        reducer.fit(X[fit_indices])
        return reducer.transform(X)
    return reducer.fit_transform(X)


def plot_split_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    splits: np.ndarray,
    *,
    panel_label: str,
    title: str,
) -> tuple[int, int]:
    train_mask = splits == "train"
    test_mask = splits == "test"
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    ax.scatter(
        embedding[train_mask, 0],
        embedding[train_mask, 1],
        s=POINT_SIZE,
        c=SPLIT_COLORS["train"],
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        embedding[test_mask, 0],
        embedding[test_mask, 1],
        s=POINT_SIZE,
        c=SPLIT_COLORS["test"],
        alpha=0.75,
        linewidths=0.2,
        edgecolors="white",
        rasterized=True,
        zorder=2,
    )

    add_panel_header(ax, panel_label, title)
    ax.set_xlabel("UMAP 1", fontsize=6)
    ax.set_ylabel("UMAP 2", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    return n_train, n_test


def add_split_legend(
    ax: plt.Axes,
    n_train: int | None = None,
    n_test: int | None = None,
    *,
    show_counts: bool = True,
    marker_size: float | None = None,
    framed: bool = False,
    legend_anchor: tuple[float, float] = (1.06, 1.0),
    legend_loc: str = "upper left",
    legend_fontsize: float = LEGEND_FONTSIZE,
    legend_fontweight: str = "bold",
) -> None:
    ms = marker_size if marker_size is not None else POINT_SIZE**0.5
    train_label = f"Train (n={n_train:,})" if show_counts and n_train is not None else "Train"
    test_label = f"Test (n={n_test:,})" if show_counts and n_test is not None else "Test"
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SPLIT_COLORS["train"],
            markersize=ms,
            alpha=0.8,
            label=train_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SPLIT_COLORS["test"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=ms,
            alpha=0.95,
            label=test_label,
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=framed,
        shadow=False,
        fancybox=False,
        handletextpad=0.55,
        borderpad=0.6,
        labelspacing=0.55,
        borderaxespad=0.0,
        prop={"size": legend_fontsize, "weight": legend_fontweight},
    )
    if framed:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#BFBFBF")
        frame.set_linewidth(0.8)
        frame.set_alpha(1.0)


def save_figure(
    fig: plt.Figure,
    out_prefix: Path,
    *,
    top: float = 0.78,
    right: float = 0.72,
) -> None:
    fig.subplots_adjust(left=0.12, right=right, top=top, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04)


def save_figure_wide(fig: plt.Figure, out_prefix: Path) -> None:
    fig.subplots_adjust(left=0.06, right=0.88, top=0.78, bottom=0.14, wspace=0.28)
    for ext in ("png", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04)


def merge_coords_with_split(
    coords_df: pd.DataFrame,
    split_csv: Path,
    id_col: str,
) -> pd.DataFrame:
    if "cluster_id" in coords_df.columns:
        return coords_df
    split_df = pd.read_csv(split_csv)
    return coords_df.merge(
        split_df[[id_col, "cluster_id"]],
        on=id_col,
        how="left",
    )


def print_split_diagnostics(df: pd.DataFrame) -> None:
    """Summarize why train/test overlap in UMAP despite cluster-pure split."""
    from sklearn.neighbors import NearestNeighbors

    splits = df["split"].astype(str).to_numpy()
    xy = df[["umap1", "umap2"]].values.astype(np.float64)
    is_test = splits == "test"

    n_test_clusters = df.loc[is_test, "cluster_id"].nunique() if "cluster_id" in df.columns else None
    n_train_clusters = df.loc[~is_test, "cluster_id"].nunique() if "cluster_id" in df.columns else None

    nn = NearestNeighbors(n_neighbors=15, metric="euclidean").fit(xy)
    _, idx = nn.kneighbors(xy)
    test_frac = (splits[idx] == "test").mean(axis=1)

    print("--- Split diagnostics (UMAP 2D) ---")
    if n_test_clusters is not None:
        print(f"Test clusters (all-test): {n_test_clusters}; Train clusters: {n_train_clusters}")
    print(
        f"Train points: {int((~is_test).sum()):,}; "
        f"mean fraction of 15-NN that are test = {test_frac[~is_test].mean():.3f}"
    )
    print(
        f"Test points: {int(is_test.sum()):,}; "
        f"mean fraction of 15-NN that are test = {test_frac[is_test].mean():.3f}"
    )
    print(
        "FAISS split holds out entire clusters (test clusters are spread for diversity, "
        "not spatial isolation in UMAP)."
    )


def plot_contrast_panel(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    suptitle: str | None = None,
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> tuple[int, int]:
    """Faint train background + highlighted test (publication-style by default)."""
    apply_nature_style()
    xy = df[["umap1", "umap2"]].values
    splits = df["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    test_mask = splits == "test"

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0))
    ax.scatter(
        xy[train_mask, 0],
        xy[train_mask, 1],
        s=POINT_SIZE - 1,
        c=SPLIT_COLORS["train"],
        alpha=0.06,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        xy[test_mask, 0],
        xy[test_mask, 1],
        s=POINT_SIZE + 4,
        c=SPLIT_COLORS["test"],
        alpha=0.95,
        linewidths=0.25,
        edgecolors="white",
        rasterized=True,
        zorder=2,
    )
    if panel_label or panel_title:
        add_panel_header(ax, panel_label or "", panel_title)
    ax.set_xlabel("UMAP 1", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    add_split_legend(
        ax,
        show_counts=False,
        marker_size=LEGEND_MARKER_SIZE,
        framed=True,
        legend_anchor=(0.98, 0.98),
        legend_loc="upper right",
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=8, y=1.02)
    save_figure(
        fig,
        out_prefix,
        top=0.94 if not suptitle and not panel_title else 0.78,
        right=0.98,
    )
    plt.close(fig)
    return n_train, n_test


def plot_enhanced_panels(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    suptitle: str,
) -> None:
    """Three-panel figure to make cluster-holdout structure easier to see."""
    apply_nature_style()
    xy = df[["umap1", "umap2"]].values
    splits = df["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    test_mask = splits == "test"
    has_cluster = "cluster_id" in df.columns
    n_test_clusters = (
        int(df.loc[test_mask, "cluster_id"].nunique()) if has_cluster and test_mask.any() else None
    )
    test_only_title = (
        f"Test only\n({n_test_clusters} FAISS clusters)"
        if n_test_clusters is not None
        else "Test only"
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    panel_specs = [
        ("a", test_only_title),
        ("b", "Test by cluster ID"),
        ("c", "Train (faint) + test"),
    ]

    # (a) test only
    ax = axes[0]
    ax.scatter(
        xy[test_mask, 0],
        xy[test_mask, 1],
        s=POINT_SIZE + 2,
        c=SPLIT_COLORS["test"],
        alpha=0.85,
        linewidths=0.2,
        edgecolors="white",
        rasterized=True,
    )
    add_panel_header(ax, panel_specs[0][0], panel_specs[0][1])
    ax.set_xlabel("UMAP 1", fontsize=6)
    ax.set_ylabel("UMAP 2", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

    # (b) test colored by cluster_id
    ax = axes[1]
    if has_cluster:
        test_df = df.loc[test_mask]
        cluster_ids = test_df["cluster_id"].astype(int).to_numpy()
        uniq = np.sort(np.unique(cluster_ids))
        cmap = plt.get_cmap("tab20")
        cid_to_color = {cid: cmap(i % 20) for i, cid in enumerate(uniq)}
        colors = [cid_to_color[c] for c in cluster_ids]
        ax.scatter(
            test_df["umap1"],
            test_df["umap2"],
            s=POINT_SIZE + 2,
            c=colors,
            alpha=0.9,
            linewidths=0.15,
            edgecolors="white",
            rasterized=True,
        )
    add_panel_header(ax, panel_specs[1][0], panel_specs[1][1])
    ax.set_xlabel("UMAP 1", fontsize=6)
    ax.set_ylabel("UMAP 2", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

    # (c) high-contrast overlay (same style as plot_contrast_panel)
    ax = axes[2]
    ax.scatter(
        xy[train_mask, 0],
        xy[train_mask, 1],
        s=POINT_SIZE - 1,
        c=SPLIT_COLORS["train"],
        alpha=0.06,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        xy[test_mask, 0],
        xy[test_mask, 1],
        s=POINT_SIZE + 4,
        c=SPLIT_COLORS["test"],
        alpha=0.95,
        linewidths=0.25,
        edgecolors="white",
        rasterized=True,
        zorder=2,
    )
    add_panel_header(ax, panel_specs[2][0], panel_specs[2][1])
    ax.set_xlabel("UMAP 1", fontsize=6)
    ax.set_ylabel("UMAP 2", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SPLIT_COLORS["train"],
            markersize=POINT_SIZE**0.5,
            alpha=0.5,
            label=f"Train (n={n_train:,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SPLIT_COLORS["test"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=(POINT_SIZE + 4) ** 0.5,
            alpha=0.95,
            label=f"Test (n={n_test:,})",
        ),
    ]
    axes[-1].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.04, 1.0),
        bbox_transform=axes[-1].transAxes,
        frameon=False,
        fontsize=7,
    )

    fig.suptitle(suptitle, fontsize=8, y=1.02)
    enhanced_prefix = Path(str(out_prefix) + "_enhanced")
    save_figure_wide(fig, enhanced_prefix)
    plt.close(fig)
    print(f"Saved enhanced figure: {enhanced_prefix.with_suffix('.png')}")


def build_features_clustering(
    split_csv: Path,
    npz_path: Path,
    id_col: str,
    w_bits: float,
    w_meta: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-block L2 + w_bits/w_meta weighting (same as FAISS clustering)."""
    df = pd.read_csv(split_csv)
    if "split" not in df.columns:
        raise ValueError(f"{split_csv} must contain a 'split' column")
    X, valid_df, errors = build_feature_matrix_from_npz(
        df,
        id_col,
        npz_path,
        w_bits=w_bits,
        w_meta=w_meta,
    )
    if errors:
        print(f"Warning: {len(errors)} rows skipped (no fingerprint in npz)")
    # valid_df rows already match df.iloc[keep_idx] from build_feature_matrix_from_npz.
    valid_df["split"] = valid_df["split"].astype(str)
    return X, valid_df


def build_features_raw(
    split_csv: Path,
    npz_path: Path,
    id_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Native ComplexFingerprint from npz: Morgan+MACCS+RDKit bits + meta one-hot, no normalization."""
    df = pd.read_csv(split_csv)
    if "split" not in df.columns:
        raise ValueError(f"{split_csv} must contain a 'split' column")

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

    ids = df[id_col].astype(str).to_numpy()
    for i in tqdm(range(len(df)), desc="Load raw fingerprints"):
        csd = ids[i]
        j = csd_to_i.get(csd)
        if j is None:
            errors.append(f"row {i} ({csd}): no precomputed fingerprint")
            continue
        rows.append(vectors[j].astype(np.float32, copy=False))
        keep_idx.append(i)

    if not rows:
        raise RuntimeError(f"No rows matched fingerprints in {npz_path}")

    X = np.vstack(rows)
    valid_df = df.iloc[keep_idx].reset_index(drop=True)
    valid_df["metal_symbol"] = [str(metals[csd_to_i[ids[i]]]) for i in keep_idx]
    valid_df["formal_charge"] = [int(charges[csd_to_i[ids[i]]]) for i in keep_idx]
    valid_df["d_electrons"] = [int(d_elecs[csd_to_i[ids[i]]]) for i in keep_idx]
    valid_df["split"] = df.iloc[keep_idx]["split"].astype(str).to_numpy()
    if "cluster_id" in df.columns:
        valid_df["cluster_id"] = df.iloc[keep_idx]["cluster_id"].values

    print(f"Loaded raw cache: {npz_path} ({len(csd_codes):,} entries)")
    if errors:
        print(f"Warning: {len(errors)} rows skipped (no fingerprint in npz)")
    return X, valid_df


def save_coords_csv(
    path: Path,
    valid_df: pd.DataFrame,
    embedding: np.ndarray,
    id_col: str,
) -> None:
    out = valid_df[[id_col, "split"]].copy()
    if "cluster_id" in valid_df.columns:
        out["cluster_id"] = valid_df["cluster_id"]
    if "metal_symbol" in valid_df.columns:
        out["metal_symbol"] = valid_df["metal_symbol"]
    out["umap1"] = embedding[:, 0]
    out["umap2"] = embedding[:, 1]
    out.to_csv(path, index=False)


def replot_from_coords(
    coords_csv: Path,
    out_prefix: Path,
    *,
    panel_title: str,
) -> None:
    apply_nature_style()
    df = pd.read_csv(coords_csv)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0))
    splits = df["split"].astype(str).to_numpy()
    embedding = df[["umap1", "umap2"]].values
    n_train, n_test = plot_split_panel(
        ax,
        embedding,
        splits,
        panel_label=PANEL_LABEL,
        title=panel_title,
    )
    add_split_legend(ax, n_train, n_test)
    save_figure(fig, out_prefix)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="UMAP plot for tmQMg FAISS train/test split (ComplexFingerprint)."
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="CSV with CSD code, split, cluster_id (from split_train_test_faiss.py). Required unless --replot-from-csv.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="JSON metadata (npz path, w_bits, w_meta, seed). Default: <split-csv>.json",
    )
    parser.add_argument(
        "--fingerprints-npz",
        type=Path,
        default=None,
        help="Override precomputed fingerprints .npz path.",
    )
    parser.add_argument("--id-col", default="CSD code")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Figure/coords output prefix. Default: <split-csv-stem>_umap next to split CSV.",
    )
    parser.add_argument("--seed", type=int, default=None, help="UMAP seed (default: split json).")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument(
        "--fit-subsample",
        type=int,
        default=20000,
        help="Fit UMAP on this many stratified points; transform all. 0 = fit on full set.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use native npz vectors (no per-block L2, no w_bits/w_meta weighting).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="UMAP metric (default: cosine for clustering features, euclidean for --raw).",
    )
    parser.add_argument(
        "--replot-from-csv",
        type=Path,
        default=None,
        help="Skip UMAP; replot from saved coords CSV.",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Also write 3-panel diagnostic figure (test-only / by cluster / high contrast).",
    )
    parser.add_argument(
        "--contrast-only",
        action="store_true",
        help="Only panel C: faint train + highlighted test (skip default overlay plot).",
    )
    args = parser.parse_args()

    if args.replot_from_csv is not None:
        coords_df = pd.read_csv(args.replot_from_csv)
        if "cluster_id" not in coords_df.columns:
            if args.split_csv is None:
                parser.error(
                    "--split-csv is required when replot CSV lacks cluster_id "
                    "(to merge cluster labels)."
                )
            coords_df = merge_coords_with_split(coords_df, args.split_csv, args.id_col)
        meta = load_split_metadata(
            args.split_json
            or (args.split_csv.with_suffix(".json") if args.split_csv is not None else None)
        )
        if args.output_prefix is None:
            args.output_prefix = (
                default_output_prefix(args.split_csv)
                if args.split_csv is not None
                else args.replot_from_csv.with_suffix("")
            )
        panel_title = panel_title_raw(meta) if args.raw else panel_title_cluster(meta)
        suptitle = suptitle_for(meta, raw=args.raw)

        if args.contrast_only:
            n_train, n_test = plot_contrast_panel(
                coords_df,
                args.output_prefix,
            )
            print(f"Train: {n_train:,}; Test: {n_test:,}")
            print(f"Saved contrast figure: {args.output_prefix.with_suffix('.png')}")
        else:
            replot_from_coords(
                args.replot_from_csv,
                args.output_prefix,
                panel_title=panel_title,
            )
            print(f"Replotted: {args.output_prefix.with_suffix('.png')}")
        if args.enhanced:
            print_split_diagnostics(coords_df)
            plot_enhanced_panels(coords_df, args.output_prefix, suptitle=suptitle)
        return

    if args.split_csv is None:
        parser.error("--split-csv is required (output of split_train_test_faiss.py).")
    if not args.split_csv.is_file():
        parser.error(f"--split-csv not found: {args.split_csv}")

    if args.split_json is None:
        args.split_json = args.split_csv.with_suffix(".json")
    if args.output_prefix is None:
        args.output_prefix = default_output_prefix(args.split_csv)

    meta = load_split_metadata(args.split_json)
    panel_title = panel_title_raw(meta) if args.raw else panel_title_cluster(meta)
    suptitle = suptitle_for(meta, raw=args.raw)
    npz_path = args.fingerprints_npz or Path(
        meta.get("fingerprints_npz", "/data/jingyuan_data/tmQMg_complex_fingerprints.npz")
    )
    w_bits = float(meta.get("w_bits", 0.7))
    w_meta = float(meta.get("w_meta", 0.3))
    seed = args.seed if args.seed is not None else int(meta.get("seed", 42))
    metric = args.metric or ("euclidean" if args.raw else "cosine")

    if not npz_path.is_file():
        raise FileNotFoundError(f"Fingerprints not found: {npz_path}")

    if args.raw:
        X, valid_df = build_features_raw(args.split_csv, npz_path, args.id_col)
        print(f"Features: {X.shape[0]:,} x {X.shape[1]} (raw ComplexFingerprint, no normalization)")
    else:
        X, valid_df = build_features_clustering(
            args.split_csv,
            npz_path,
            args.id_col,
            w_bits=w_bits,
            w_meta=w_meta,
        )
        print(
            f"Features: {X.shape[0]:,} x {X.shape[1]} "
            f"(clustering L2 norm, w_bits={w_bits}, w_meta={w_meta})"
        )

    splits = valid_df["split"].astype(str).to_numpy()
    n = X.shape[0]

    fit_idx = None
    if args.fit_subsample and args.fit_subsample < n:
        fit_idx = stratified_fit_indices(splits, args.fit_subsample, seed)
        print(f"UMAP fit on {len(fit_idx):,} stratified points; transform all {n:,}")
    else:
        print(f"UMAP fit on all {n:,} points")

    print(f"UMAP metric: {metric}")
    embedding = run_umap(
        X,
        seed=seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=metric,
        fit_indices=fit_idx,
    )

    coords_csv = args.output_prefix.with_suffix(".csv")
    save_coords_csv(coords_csv, valid_df, embedding, args.id_col)

    coords_df = valid_df[[args.id_col, "split"]].copy()
    if "cluster_id" in valid_df.columns:
        coords_df["cluster_id"] = valid_df["cluster_id"]
    if "metal_symbol" in valid_df.columns:
        coords_df["metal_symbol"] = valid_df["metal_symbol"]
    coords_df["umap1"] = embedding[:, 0]
    coords_df["umap2"] = embedding[:, 1]

    if args.contrast_only:
        n_train, n_test = plot_contrast_panel(
            coords_df,
            args.output_prefix,
        )
        print(f"Train: {n_train:,}; Test: {n_test:,}")
        print(f"Saved coords: {coords_csv}")
        print(f"Saved contrast figure: {args.output_prefix.with_suffix('.png')}")
    else:
        apply_nature_style()
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.0))
        n_train, n_test = plot_split_panel(
            ax,
            embedding,
            splits,
            panel_label=PANEL_LABEL,
            title=panel_title,
        )
        add_split_legend(ax, n_train, n_test)
        save_figure(fig, args.output_prefix)
        plt.close(fig)

        print(f"Train: {n_train:,}; Test: {n_test:,}")
        print(f"Saved coords: {coords_csv}")
        print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
        print(f"Saved figure: {args.output_prefix.with_suffix('.pdf')}")

    if args.enhanced:
        print_split_diagnostics(coords_df)
        plot_enhanced_panels(coords_df, args.output_prefix, suptitle=suptitle)


if __name__ == "__main__":
    main()
