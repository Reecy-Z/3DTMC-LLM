"""Leave-one-ligand-out (LOLO) split for Vaska OOD (vaskas-space/data.lmdb)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

LIGAND_SLOT_FIELDS = ("ligand_a1", "ligand_a2", "ligand_b", "ligand_c")

# 26 unique ligands in vaskas-space/data.lmdb (1947 samples in the reference release).
LIGANDS = (
    "bromide",
    "chloride",
    "dft-asme3",
    "dft-cch",
    "dft-cn",
    "dft-co",
    "dft-hicn",
    "dft-iacn",
    "dft-icn",
    "dft-ime",
    "dft-itcn",
    "dft-iz",
    "dft-nme3",
    "dft-no2",
    "dft-oh",
    "dft-oxaz",
    "dft-pet3",
    "dft-phos",
    "dft-pme3",
    "dft-pph3",
    "dft-py",
    "dft-pyz",
    "dft-sh",
    "dft-sime",
    "fluoride",
    "iodide",
)


def ligand_dirname(ligand: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_+-]+", "_", ligand.strip())
    return f"ligand_{safe}"


def sample_ligands(sample: dict) -> Set[str]:
    return {sample[f] for f in LIGAND_SLOT_FIELDS if sample.get(f)}


def sample_has_ligand(sample: dict, ligand: str) -> bool:
    return ligand in sample_ligands(sample)


def summarize_ligand_presence(samples: List[dict]) -> Dict[str, int]:
    """Count samples where each ligand appears in any of the four slots."""
    counts: Counter[str] = Counter()
    for s in samples:
        for lig in sample_ligands(s):
            counts[lig] += 1
    return dict(counts)


def lolo_split_by_ligand(
    samples: List[dict],
    holdout_ligand: str,
) -> Tuple[List[dict], List[dict]]:
    """Hold out samples containing ``holdout_ligand`` in any slot; rest → train."""
    if holdout_ligand not in LIGANDS:
        raise ValueError(f"Unknown holdout_ligand={holdout_ligand!r}; expected one of {LIGANDS}")

    ood_test = [s for s in samples if sample_has_ligand(s, holdout_ligand)]
    train = [s for s in samples if not sample_has_ligand(s, holdout_ligand)]
    if not train:
        raise RuntimeError(f"No train samples left after holding out ligand={holdout_ligand!r}")
    if not ood_test:
        raise RuntimeError(f"No OOD test samples for holdout ligand={holdout_ligand!r}")
    return train, ood_test
