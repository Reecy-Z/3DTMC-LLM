"""Property LMDB dataset filtered by cluster split CSV or split directory (OOD)."""
from __future__ import annotations

import os
import pickle
from typing import Optional, Set

import lmdb
import numpy as np

from Property import PROPERTY_CONFIG
from utils import (
    _atoms_coords_remove_h_center,
    _lmdb_env_kwargs,
    format_instruction_field,
    tokenize_generation_sample_object_ref,
)

from OOD.cluster_split import csd_from_lmdb_key, load_csd_codes_for_split, summarize_split_source


class TmQMgClusterSplitDataset:
    """Same sample format as TmQMgSingleTokenUnimolDataset; filter LMDB keys by CSV split."""

    def __init__(
        self,
        lmdb_paths,
        split_csv: str,
        split_name: str,
        tokenizer,
        max_samples=None,
        property_key="dipole_moment",
        instruction=None,
        use_polished_description=False,
        allowed_csd_codes: Optional[Set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.property_key = property_key
        self.split_name = split_name.strip().lower()
        self.use_polished_description = use_polished_description
        pcfg = PROPERTY_CONFIG.get(property_key, PROPERTY_CONFIG["dipole_moment"])
        if instruction is not None:
            self.instruction = instruction
        elif use_polished_description:
            self.instruction = pcfg.get("instruction_description") or pcfg["instruction_smiles"]
        else:
            self.instruction = pcfg.get("instruction_smiles") or pcfg.get("instruction_description")

        if allowed_csd_codes is None:
            allowed_csd_codes = load_csd_codes_for_split(split_csv, self.split_name)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"[Dataset-OOD] split source: {split_csv} | "
                    f"labels: {summarize_split_source(split_csv)}"
                )
        self.allowed_csd_codes = allowed_csd_codes

        self.key_index = []
        self._envs = {}
        nan_skipped = 0
        smiles_skipped = 0
        split_skipped = 0
        stop = False
        for p in lmdb_paths:
            if not p or not os.path.exists(p):
                continue
            env = lmdb.open(p, **_lmdb_env_kwargs())
            try:
                with env.begin() as txn:
                    for key_bytes, value in txn.cursor():
                        csd = csd_from_lmdb_key(key_bytes)
                        if csd not in self.allowed_csd_codes:
                            split_skipped += 1
                            continue
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
            raise RuntimeError(
                f"No valid samples for split={self.split_name!r}: lmdb={lmdb_paths}, csv={split_csv}"
            )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if split_skipped > 0:
                print(f"[Dataset-OOD] Skipped {split_skipped} LMDB keys not in split={self.split_name}")
            if nan_skipped > 0:
                print(f"[Dataset-OOD] Skipped {nan_skipped} samples with NaN {property_key}")
            if smiles_skipped > 0:
                print(f"[Dataset-OOD] Skipped {smiles_skipped} samples (missing or empty SMILES)")
            print(
                f"[Dataset-OOD] property={property_key}, split={self.split_name}, "
                f"{len(self.key_index)} samples; mode=SMILES+3D (cluster k200)"
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
        smiles = format_instruction_field(data.get("smiles"))
        response = str(data[self.property_key])
        atoms = data["atoms"]
        if isinstance(atoms, np.ndarray):
            atoms = atoms.tolist()
        atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        atoms, coords = _atoms_coords_remove_h_center(atoms, coords)
        if self.use_polished_description and self.property_key == "homo_lumo_gap" and "polished_description" in data:
            desc = data.get("polished_description")
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
            if desc is not None:
                desc = str(desc).strip()
            if desc:
                user_content = f"{desc}\n{self.instruction} {smiles}"
            else:
                user_content = f"{self.instruction} {smiles}"
        else:
            user_content = f"{self.instruction} {smiles}"
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}
