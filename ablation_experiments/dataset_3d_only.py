"""Property LMDB dataset: prompt uses 3D structure slot only (no SMILES text)."""
from __future__ import annotations

import os
import pickle

import lmdb
import numpy as np

from Property import PROPERTY_CONFIG
from utils import (
    _atoms_coords_remove_h_center,
    _lmdb_env_kwargs,
    format_instruction_field,
    tokenize_generation_sample_object_ref,
)

# Ablation: no SMILES in the user message; structure is injected via object_ref 3D tokens.
INSTRUCTION_3D_ONLY = {
    "dipole_moment": (
        "What is the dipole moment (in Debye) of this transition metal complex? "
        "Given the 3D structure only, respond with the numerical value only:"
    ),
    "polarisability": (
        "What is the polarisability (in Bohr^3) of this transition metal complex? "
        "Given the 3D structure only, respond with the numerical value only:"
    ),
    "homo_lumo_gap": (
        "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? "
        "Given the 3D structure only, respond with the numerical value only:"
    ),
}


class TmQMg3DOnlyUnimolDataset:
    """Same LMDB filtering as TmQMgSingleTokenUnimolDataset; user text is instruction only."""

    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_samples=None,
        property_key="dipole_moment",
        instruction=None,
    ):
        self.tokenizer = tokenizer
        self.property_key = property_key
        if property_key not in INSTRUCTION_3D_ONLY:
            raise ValueError(f"Unknown property_key={property_key!r}")
        self.instruction = instruction or INSTRUCTION_3D_ONLY[property_key]
        self.key_index = []
        self._envs = {}
        nan_skipped = 0
        smiles_skipped = 0
        stop = False
        for p in lmdb_paths:
            if not p or not os.path.exists(p):
                continue
            env = lmdb.open(p, **_lmdb_env_kwargs())
            try:
                with env.begin() as txn:
                    for key_bytes, value in txn.cursor():
                        d = pickle.loads(value)
                        if "atoms" not in d or "coordinates" not in d:
                            continue
                        if "smiles" not in d:
                            continue
                        if not format_instruction_field(d.get("smiles")):
                            smiles_skipped += 1
                            continue
                        if property_key not in d:
                            continue
                        try:
                            val = float(d[property_key])
                            if np.isnan(val):
                                nan_skipped += 1
                                continue
                        except (ValueError, TypeError):
                            nan_skipped += 1
                            continue
                        self.key_index.append((p, key_bytes))
                        if max_samples is not None and len(self.key_index) >= max_samples:
                            stop = True
                            break
                    if stop:
                        break
            finally:
                env.close()
            if stop:
                break
        if not self.key_index and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(f"No valid samples found: {lmdb_paths}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset-3D-only] Skipped {nan_skipped} samples with NaN {property_key}")
        if smiles_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset-3D-only] Skipped {smiles_skipped} samples (missing SMILES in LMDB; not used in prompt)")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            pcfg = PROPERTY_CONFIG.get(property_key, {})
            print(
                f"[Dataset-3D-only] property={property_key}, {len(self.key_index)} samples; "
                f"mode=3D-only (no SMILES in text; {pcfg.get('unit', '')})"
            )

    def __len__(self):
        return len(self.key_index)

    def __getitem__(self, idx):
        lmdb_path, key_bytes = self.key_index[idx]
        env = self._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
            self._envs[lmdb_path] = env
        with env.begin() as txn:
            raw = txn.get(key_bytes)
            if raw is None:
                raise KeyError(f"LMDB key not found: {key_bytes!r}")
            data = pickle.loads(raw)
        response = str(data[self.property_key])
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        user_content = self.instruction.strip()
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}
