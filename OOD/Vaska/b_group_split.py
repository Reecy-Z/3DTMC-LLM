"""Leave-one-b_group-out split for Vaska OOD (vaskas-space/data.lmdb)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

B_GROUPS = (
    "linear_pseudohalides",
    "halides",
    "chalcogen_donors",
    "nitro_nitrite",
    "alkynyl",
)


def b_group_dirname(b_group: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", b_group.strip())
    return f"bgroup_{safe}"


def summarize_b_groups(samples: List[dict]) -> Dict[str, int]:
    return dict(Counter(s.get("b_group", "<missing>") for s in samples))


def lobo_split_by_b_group(
    samples: List[dict],
    holdout_b_group: str,
) -> Tuple[List[dict], List[dict]]:
    """Hold out one b_group for OOD test; all other b_groups go to training."""
    if holdout_b_group not in B_GROUPS:
        raise ValueError(f"Unknown holdout_b_group={holdout_b_group!r}; expected one of {B_GROUPS}")

    ood_test = [s for s in samples if s.get("b_group") == holdout_b_group]
    train = [s for s in samples if s.get("b_group") and s.get("b_group") != holdout_b_group]
    if not train:
        raise RuntimeError(f"No train samples left after holding out {holdout_b_group!r}")
    if not ood_test:
        raise RuntimeError(f"No OOD test samples for holdout b_group={holdout_b_group!r}")
    return train, ood_test
