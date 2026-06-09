"""LMDB datasets for description, property, Vaska, and NiComplex tasks."""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Optional

import lmdb
import numpy as np
from torch.utils.data import Dataset

from task_registry import (
    INSTRUCTION_DESCRIPTION,
    INSTRUCTION_DESCRIPTION_3D_ONLY,
    NI_INSTRUCTION,
    VASKA_INSTRUCTION,
    get_task,
    instruction_3d_only_dict,
    property_config_dict,
    resolve_instruction,
    resolve_user_content,
)
from utils import (
    MAX_SEQ_LENGTH,
    OBJECT_REF_CHAT_SEP,
    _atoms_coords_remove_h_center,
    _lmdb_env_kwargs,
    format_instruction_field,
    read_lmdb,
    tokenize_generation_sample_object_ref,
)

# Re-export registry constants for backward-compatible imports.
PROPERTY_CONFIG = property_config_dict()
INSTRUCTION_3D_ONLY = instruction_3d_only_dict()
RESPONSE_KEY = "polished_description"
CHAT_SEP = OBJECT_REF_CHAT_SEP


def _response_text(data: dict, response_key: str = RESPONSE_KEY) -> str | None:
    text = data.get(response_key)
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    return str(text).strip() if isinstance(text, str) else str(text)


def _atoms_coords_from_record(data: dict) -> tuple[list[str], np.ndarray]:
    atoms = data["atoms"]
    if isinstance(atoms, np.ndarray):
        atoms = atoms.tolist()
    atoms = [str(a) if not hasattr(a, "item") else str(a.item()) for a in atoms]
    coords = np.asarray(data["coordinates"], dtype=np.float32)
    if coords.ndim == 3:
        coords = coords[0]
    return _atoms_coords_remove_h_center(atoms, coords)


def _read_lmdb_pickle_list(
    lmdb_path: str,
    *,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
    log_prefix: str = "LMDB",
) -> list[dict[str, Any]]:
    """Read pickle-serialized dict records from an LMDB (Vaska / NiComplex style)."""
    import tqdm as _tqdm

    env = lmdb.open(lmdb_path, **_lmdb_env_kwargs())
    txn = env.begin()
    cursor = txn.cursor()
    data_list: list[dict[str, Any]] = []
    stat = env.stat()
    total = min(max_samples, stat["entries"]) if max_samples else stat["entries"]
    desc = f"LMDB {os.path.basename(lmdb_path)}"
    rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
    it = (
        _tqdm.tqdm(cursor, total=total, desc=desc, unit="samples", mininterval=1.0)
        if (show_progress and rank0)
        else cursor
    )
    bad = 0
    for count, (_, value) in enumerate(it):
        try:
            obj = pickle.loads(value)
        except Exception:
            bad += 1
            continue
        data_list.append(obj)
        if max_samples and (count + 1) >= max_samples:
            break
    env.close()
    if bad > 0 and rank0:
        print(f"[{log_prefix}] skipped {bad} bad entries")
    return data_list


def read_vaska_lmdb(lmdb_path, max_samples=None, show_progress=True):
    """LMDB reader for Vaska dataset."""
    return _read_lmdb_pickle_list(
        lmdb_path,
        max_samples=max_samples,
        show_progress=show_progress,
        log_prefix="Vaska-LMDB",
    )


def read_nicomplex_lmdb(lmdb_path, max_samples=None, show_progress=True):
    """LMDB reader for NiComplex dataset."""
    return _read_lmdb_pickle_list(
        lmdb_path,
        max_samples=max_samples,
        show_progress=show_progress,
        log_prefix="Stage3_NiComplex-LMDB",
    )


def load_merged_valid_nicomplex_records(lmdb_paths, local_rank=0):
    """Merge NiComplex LMDBs; return records with valid ddG."""
    all_raw: list[dict] = []
    for p in lmdb_paths:
        if not os.path.exists(p):
            if local_rank == 0:
                print(f"[NiComplex] skip missing LMDB: {p}")
            continue
        all_raw.extend(read_nicomplex_lmdb(p, max_samples=None))
    merged_valid: list[dict] = []
    nan_skipped = 0
    for d in all_raw:
        if "atoms" not in d or "coordinates" not in d or "ddG" not in d:
            continue
        try:
            val = float(d["ddG"])
            if np.isnan(val):
                nan_skipped += 1
                continue
        except (ValueError, TypeError):
            nan_skipped += 1
            continue
        merged_valid.append(d)
    return merged_valid, nan_skipped


class TmQMEnrichedLMDBDataset(Dataset):
    """LMDB: instruction + SMILES + atoms + coordinates -> polished_description."""

    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        max_samples=None,
        instruction=None,
        include_smiles=True,
        response_key=RESPONSE_KEY,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction or INSTRUCTION_DESCRIPTION
        self.include_smiles = include_smiles
        self.response_key = response_key
        self._prompt_mode = "single_token" if include_smiles else "3d_only"
        self.samples = []
        for path in lmdb_paths:
            if not os.path.exists(path):
                continue
            raw = read_lmdb(path, max_samples=max_samples)
            for data in raw:
                if "atoms" not in data or "coordinates" not in data or "smiles" not in data:
                    continue
                if _response_text(data, response_key) is None:
                    continue
                self.samples.append(data)
        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            raise RuntimeError(
                f"No valid samples (need atoms, coordinates, smiles, {response_key}): {lmdb_paths}"
            )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            mode = "instruction+SMILES" if self.include_smiles else "3D-only (no SMILES)"
            print(f"[Description-LMDB] {len(self.samples)} samples; response={response_key}; mode={mode}")

    def __len__(self):
        return len(self.samples)

    def _user_content(self, data: dict) -> str:
        return resolve_user_content(
            "description",
            data,
            mode=self._prompt_mode,
            instruction_override=self.instruction,
        )

    def __getitem__(self, idx):
        data = self.samples[idx]
        response = _response_text(data, self.response_key) or ""
        atoms, coords = _atoms_coords_from_record(data)
        user_content = self._user_content(data)
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": response}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
        prefix_len = min(prefix_len, len(enc["input_ids"]) - 1) if len(enc["input_ids"]) > 1 else 0
        response_ids = enc["input_ids"][prefix_len:]
        before_3d_str = prefix_str.split(CHAT_SEP, 1)[0]
        after_3d_str = CHAT_SEP + prefix_str.split(CHAT_SEP, 1)[1]
        before_3d_enc = self.tokenizer(before_3d_str, return_tensors=None, truncation=True, max_length=self.max_length)
        after_3d_enc = self.tokenizer(after_3d_str, return_tensors=None, truncation=True, max_length=self.max_length)
        return {
            "sample_type": "lmdb",
            "sample_idx": idx,
            "before_3d_ids": before_3d_enc["input_ids"],
            "after_3d_ids": after_3d_enc["input_ids"],
            "response_ids": response_ids,
            "atoms": atoms,
            "coordinates": coords,
        }


class TmQMEnriched3DOnlyDataset(TmQMEnrichedLMDBDataset):
    """Description LMDB with 3D slot only (no SMILES in the user prompt)."""

    def __init__(self, lmdb_paths, tokenizer, max_length=MAX_SEQ_LENGTH, max_samples=None):
        super().__init__(
            lmdb_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            instruction=INSTRUCTION_DESCRIPTION_3D_ONLY,
            include_smiles=False,
        )


class TmQMEnrichedTokenizedDataset(Dataset):
    """Same filtering as TmQMEnrichedLMDBDataset; uses tokenize_generation_sample_object_ref."""

    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        max_samples=None,
        instruction=None,
        include_smiles=True,
        response_key=RESPONSE_KEY,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction or INSTRUCTION_DESCRIPTION
        self.include_smiles = include_smiles
        self.response_key = response_key
        self._prompt_mode = "single_token" if include_smiles else "3d_only"
        self.key_index = []
        self._envs = {}
        stop = False
        for path in lmdb_paths:
            if not path or not os.path.exists(path):
                continue
            env = lmdb.open(path, **_lmdb_env_kwargs())
            try:
                with env.begin() as txn:
                    for key_bytes, value in txn.cursor():
                        data = pickle.loads(value)
                        if "atoms" not in data or "coordinates" not in data or "smiles" not in data:
                            continue
                        if _response_text(data, response_key) is None:
                            continue
                        self.key_index.append((path, key_bytes))
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
            raise RuntimeError(f"No valid description samples found: {lmdb_paths}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            mode = "instruction+SMILES" if self.include_smiles else "3D-only (no SMILES)"
            print(f"[Description-Tokenized] {len(self.key_index)} samples; response={response_key}; mode={mode}")

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
        response = _response_text(data, self.response_key) or ""
        atoms, coords = _atoms_coords_from_record(data)
        user_content = resolve_user_content(
            "description",
            data,
            mode=self._prompt_mode,
            instruction_override=self.instruction,
        )
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords, "sample_idx": idx}


class JsonQADataset(Dataset):
    """JSON Q&A: question -> answer; no 3D."""

    def __init__(self, json_paths, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for path in json_paths:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        question = item.get("question")
                        answer = item.get("answer")
                        if question is not None and answer is not None and str(question).strip() and str(answer).strip():
                            self.samples.append({"question": str(question).strip(), "answer": str(answer).strip()})
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[JsonQADataset] {len(self.samples)} text-only Q&A samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item["question"]
        answer = item["answer"]
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(full_text, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_enc = self.tokenizer(prefix_str, return_tensors=None, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])
        prefix_len = min(prefix_len, len(enc["input_ids"]) - 1) if len(enc["input_ids"]) > 1 else 0
        response_ids = enc["input_ids"][prefix_len:]
        return {
            "sample_type": "text",
            "sample_idx": -1,
            "before_3d_ids": prefix_enc["input_ids"][:prefix_len] if prefix_len > 0 else prefix_enc["input_ids"],
            "after_3d_ids": [],
            "response_ids": response_ids,
            "atoms": [],
            "coordinates": [],
        }


class TmQMgSingleTokenUnimolDataset(Dataset):
    """Property LMDB: instruction + SMILES text; structure via object_ref 3D slot."""

    def __init__(
        self,
        lmdb_paths,
        tokenizer,
        max_samples=None,
        property_key="dipole_moment",
        instruction=None,
        use_polished_description=False,
    ):
        self.tokenizer = tokenizer
        self.property_key = property_key
        self.task_name = property_key
        self.use_polished_description = use_polished_description
        pcfg = PROPERTY_CONFIG.get(property_key, PROPERTY_CONFIG["dipole_moment"])
        if instruction is not None:
            self.instruction = instruction
        else:
            self.instruction = resolve_instruction(
                property_key,
                mode="single_token",
                use_polished_description=use_polished_description,
            )
        self.key_index = []
        self._envs = {}
        nan_skipped = 0
        smiles_skipped = 0
        stop = False
        for p in lmdb_paths:
            if not os.path.exists(p):
                continue
            env = lmdb.open(p, **_lmdb_env_kwargs())
            try:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key_bytes, value in cursor:
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
            print(f"[Dataset] Skipped {nan_skipped} samples with NaN {property_key}")
        if smiles_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset] Skipped {smiles_skipped} samples (missing or empty SMILES)")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[Dataset] property={property_key}, {len(self.key_index)} samples; mode=GENERATION (with SMILES)")

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
        atoms, coords = _atoms_coords_from_record(data)
        user_content = resolve_user_content(
            self.task_name,
            data,
            mode="single_token",
            use_polished_description=self.use_polished_description,
            instruction_override=self.instruction,
        )
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}


class TmQMg3DOnlyUnimolDataset(Dataset):
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
        self.task_name = property_key
        if property_key not in INSTRUCTION_3D_ONLY:
            raise ValueError(f"Unknown property_key={property_key!r}")
        self.instruction = instruction or resolve_instruction(property_key, mode="3d_only")
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
        atoms, coords = _atoms_coords_from_record(data)
        user_content = resolve_user_content(
            self.task_name,
            data,
            mode="3d_only",
            instruction_override=self.instruction,
        )
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}


class VaskaComplexDataset(Dataset):
    """Vaska barrier: 3D block + SMILES in the chat prompt."""

    def __init__(
        self,
        lmdb_paths=None,
        tokenizer=None,
        max_samples=None,
        instruction=None,
        samples=None,
    ):
        self.tokenizer = tokenizer
        self.instruction = instruction or VASKA_INSTRUCTION
        self.property_key = get_task("vaska_barrier").target_key

        self.samples: list[dict] = []
        nan_skipped = 0
        smiles_skipped = 0
        if samples is not None:
            data_iter = samples
        else:
            data_iter = []
            for p in lmdb_paths or []:
                if not os.path.exists(p):
                    continue
                data_iter.extend(read_vaska_lmdb(p, max_samples=max_samples))

        for d in data_iter:
            if "atoms" not in d or "coordinates" not in d or self.property_key not in d:
                continue
            try:
                val = float(d[self.property_key])
                if np.isnan(val):
                    nan_skipped += 1
                    continue
            except (ValueError, TypeError):
                nan_skipped += 1
                continue
            if not format_instruction_field(d.get("smiles")):
                smiles_skipped += 1
                continue
            self.samples.append(d)

        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            src = lmdb_paths if samples is None else "provided samples"
            raise RuntimeError(f"No valid samples: {src}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] Skipped {nan_skipped} samples with NaN {self.property_key}")
        if smiles_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] Skipped {smiles_skipped} samples (missing SMILES)")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[VaskaDataset] property={self.property_key}, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        response = str(float(data[self.property_key]))
        atoms, coords = _atoms_coords_from_record(data)
        user_content = resolve_user_content(
            "vaska_barrier",
            data,
            mode="single_token",
            instruction_override=self.instruction,
        )
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, response)
        return {**ids, "atoms": atoms, "coordinates": coords}


class NiComplexDDGDataset(Dataset):
    """Ni(III) ΔΔG: templated instruction ({smiles}, {temp}) + 3D block."""

    def __init__(
        self,
        lmdb_paths=None,
        tokenizer=None,
        max_length=512,
        max_samples=None,
        prop_mean=None,
        prop_std=None,
        instruction=None,
        samples=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prop_mean = prop_mean
        self.prop_std = prop_std
        self.property_key = get_task("nicomplex_ddg").target_key
        self.instruction = instruction or NI_INSTRUCTION

        self.samples: list[dict] = []
        nan_skipped = 0
        if samples is not None:
            data_iter = samples
        else:
            data_iter = []
            for p in lmdb_paths or []:
                if not os.path.exists(p):
                    continue
                data_iter.extend(read_nicomplex_lmdb(p, max_samples=max_samples))

        for d in data_iter:
            if "atoms" not in d or "coordinates" not in d or self.property_key not in d:
                continue
            try:
                val = float(d[self.property_key])
                if np.isnan(val):
                    nan_skipped += 1
                    continue
            except (ValueError, TypeError):
                nan_skipped += 1
                continue
            self.samples.append(d)

        if not self.samples and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            src = lmdb_paths if samples is None else "provided samples"
            raise RuntimeError(f"No valid samples: {src}")
        if nan_skipped > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[NiComplexDataset] Skipped {nan_skipped} samples with NaN {self.property_key}")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"[NiComplexDataset] property={self.property_key}, {len(self.samples)} samples; "
                f"mode=GENERATION"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        atoms, coords = _atoms_coords_from_record(d)
        target_text = f"{float(d[self.property_key]):.6f}"
        user_content = resolve_user_content(
            "nicomplex_ddg",
            d,
            mode="single_token",
            instruction_override=self.instruction,
        )
        ids = tokenize_generation_sample_object_ref(self.tokenizer, user_content, target_text)
        return {**ids, "atoms": atoms, "coordinates": coords}


atoms_coords_from_record = _atoms_coords_from_record

__all__ = [
    "atoms_coords_from_record",
    "CHAT_SEP",
    "INSTRUCTION_3D_ONLY",
    "INSTRUCTION_DESCRIPTION",
    "INSTRUCTION_DESCRIPTION_3D_ONLY",
    "NI_INSTRUCTION",
    "PROPERTY_CONFIG",
    "RESPONSE_KEY",
    "VASKA_INSTRUCTION",
    "JsonQADataset",
    "NiComplexDDGDataset",
    "TmQMEnriched3DOnlyDataset",
    "TmQMEnrichedLMDBDataset",
    "TmQMEnrichedTokenizedDataset",
    "TmQMg3DOnlyUnimolDataset",
    "TmQMgSingleTokenUnimolDataset",
    "VaskaComplexDataset",
    "load_merged_valid_nicomplex_records",
    "read_nicomplex_lmdb",
    "read_vaska_lmdb",
]
