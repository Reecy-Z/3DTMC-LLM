#!/usr/bin/env python3
"""Concatenated complex fingerprint: Morgan + MACCS + RDKit topo + metal features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

RDLogger.DisableLog("rdApp.*")

PT = Chem.GetPeriodicTable()

# d-block transition metals in tmQM (30): 3d + 4d + La + 5d (Hf–Hg); no lanthanides Ce–Lu
D_BLOCK_METAL_Z = list(range(21, 31)) + list(range(39, 49)) + [57] + list(range(72, 81))
TRANSITION_METAL_Z = D_BLOCK_METAL_Z
METAL_SYMBOLS = [PT.GetElementSymbol(z) for z in D_BLOCK_METAL_Z]
METAL_INDEX = {sym: i for i, sym in enumerate(METAL_SYMBOLS)}

MORGAN_BITS = 2048
MACCS_BITS = 167
RDKIT_BITS = 2048
BIT_DIM = MORGAN_BITS + MACCS_BITS + RDKIT_BITS

# formal charge (oxidation state) one-hot for integers in [-2, +7] (tmQM coverage >99.9%)
CHARGE_MIN = -2
CHARGE_MAX = 7
CHARGE_DIM = CHARGE_MAX - CHARGE_MIN + 1

# d-electron count one-hot for integers in [0, 12]
D_ELECTRON_MIN = 0
D_ELECTRON_MAX = 12
D_ELECTRON_DIM = D_ELECTRON_MAX - D_ELECTRON_MIN + 1

META_DIM = len(METAL_SYMBOLS) + CHARGE_DIM + D_ELECTRON_DIM
TOTAL_DIM = BIT_DIM + META_DIM

# Slice indices for per-block L2 normalization (clustering / FAISS).
SLICE_MORGAN = slice(0, MORGAN_BITS)
SLICE_MACCS = slice(MORGAN_BITS, MORGAN_BITS + MACCS_BITS)
SLICE_RDKIT = slice(MORGAN_BITS + MACCS_BITS, BIT_DIM)
SLICE_METAL = slice(BIT_DIM, BIT_DIM + len(METAL_SYMBOLS))
SLICE_CHARGE = slice(BIT_DIM + len(METAL_SYMBOLS), BIT_DIM + len(METAL_SYMBOLS) + CHARGE_DIM)
SLICE_D_ELECTRON = slice(BIT_DIM + len(METAL_SYMBOLS) + CHARGE_DIM, TOTAL_DIM)
CLUSTER_BLOCKS = (
    ("morgan", SLICE_MORGAN),
    ("maccs", SLICE_MACCS),
    ("rdkit", SLICE_RDKIT),
    ("metal", SLICE_METAL),
    ("charge", SLICE_CHARGE),
    ("d_electron", SLICE_D_ELECTRON),
)


@dataclass(frozen=True)
class ComplexFingerprint:
    vector: np.ndarray
    metal_symbol: str
    formal_charge: int
    d_electrons: int
    smiles: str

    @property
    def bit_vector(self) -> np.ndarray:
        return self.vector[:BIT_DIM]

    @property
    def meta_vector(self) -> np.ndarray:
        return self.vector[BIT_DIM:]


def get_d_electrons(metal_symbol: str, oxidation_state: int) -> int:
    z = PT.GetAtomicNumber(metal_symbol)
    n_outer = PT.GetNOuterElecs(z)
    return n_outer - oxidation_state


def find_transition_metal(mol: Chem.Mol) -> tuple[str, int] | None:
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in TRANSITION_METAL_Z:
            return atom.GetSymbol(), atom.GetFormalCharge()
    return None


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32, copy=True)
    return (vec / norm).astype(np.float32, copy=False)


def vector_for_clustering(
    fp: ComplexFingerprint,
    *,
    w_bits: float = 0.7,
    w_meta: float = 0.3,
    final_l2: bool = True,
) -> np.ndarray:
    """
    Per-block L2 normalize, scale by sqrt(block weight), concatenate for FAISS.

    Three 2D fingerprints share w_bits; metal / charge / d share w_meta.
    """
    n_fp = 3
    n_meta = 3
    parts: list[np.ndarray] = []
    for name, slc in CLUSTER_BLOCKS:
        block = fp.vector[slc].astype(np.float32, copy=False)
        block = l2_normalize(block)
        if name in ("morgan", "maccs", "rdkit"):
            block = block * np.sqrt(w_bits / n_fp, dtype=np.float32)
        else:
            block = block * np.sqrt(w_meta / n_meta, dtype=np.float32)
        parts.append(block)
    out = np.concatenate(parts)
    if final_l2:
        out = l2_normalize(out)
    return out


def _bitvect_to_array(fp) -> np.ndarray:
    arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _encode_meta(metal_symbol: str, formal_charge: int, d_electrons: int) -> np.ndarray:
    meta = np.zeros(META_DIM, dtype=np.float32)
    if metal_symbol in METAL_INDEX:
        meta[METAL_INDEX[metal_symbol]] = 1.0
    if CHARGE_MIN <= formal_charge <= CHARGE_MAX:
        charge_idx = len(METAL_SYMBOLS) + (formal_charge - CHARGE_MIN)
        meta[charge_idx] = 1.0
    if D_ELECTRON_MIN <= d_electrons <= D_ELECTRON_MAX:
        d_idx = len(METAL_SYMBOLS) + CHARGE_DIM + (d_electrons - D_ELECTRON_MIN)
        meta[d_idx] = 1.0
    return meta


def build_complex_fingerprint(smiles: str) -> ComplexFingerprint | None:
    """
    Overall complex fingerprint layout (length TOTAL_DIM = 4316):

      [0:2048)     Morgan ECFP4 (radius=2)
      [2048:2215)  MACCS keys
      [2215:4263)  RDKit topological fingerprint
      [4263:4293)  metal_symbol one-hot (30 d-block metals, tmQM set)
      [4293:4303)  formal_charge one-hot (-2 .. +7)
      [4303:4316)  d_electron count one-hot (0 .. 12)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    metal_info = find_transition_metal(mol)
    if metal_info is None:
        return None
    metal_symbol, formal_charge = metal_info
    d_electrons = get_d_electrons(metal_symbol, formal_charge)

    morgan = _bitvect_to_array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
    )
    maccs = _bitvect_to_array(MACCSkeys.GenMACCSKeys(mol))
    rdkit = _bitvect_to_array(Chem.RDKFingerprint(mol, fpSize=RDKIT_BITS))
    bits = np.concatenate([morgan, maccs, rdkit])
    meta = _encode_meta(metal_symbol, formal_charge, d_electrons)
    vector = np.concatenate([bits, meta])

    return ComplexFingerprint(
        vector=vector,
        metal_symbol=metal_symbol,
        formal_charge=formal_charge,
        d_electrons=d_electrons,
        smiles=smiles,
    )


def tanimoto_bits(a: ComplexFingerprint, b: ComplexFingerprint) -> float:
    """Tanimoto on the 2D fingerprint bit block only."""
    return _tanimoto_numpy(a.bit_vector, b.bit_vector)


def _tanimoto_numpy(x: np.ndarray, y: np.ndarray) -> float:
    inter = float(np.dot(x, y))
    union = float(np.sum(x) + np.sum(y) - inter)
    return 1.0 if union == 0 else inter / union


def cosine_full(a: ComplexFingerprint, b: ComplexFingerprint) -> float:
    """Cosine similarity on the full concatenated vector (bits + meta)."""
    va, vb = a.vector, b.vector
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 1.0 if denom == 0 else float(np.dot(va, vb) / denom)


def cosine_meta(a: ComplexFingerprint, b: ComplexFingerprint) -> float:
    """Cosine on metal / charge / d-electron block only."""
    ma, mb = a.meta_vector, b.meta_vector
    denom = float(np.linalg.norm(ma) * np.linalg.norm(mb))
    return 1.0 if denom == 0 else float(np.dot(ma, mb) / denom)


def similarity(
    a: ComplexFingerprint,
    b: ComplexFingerprint,
    *,
    w_bits: float = 0.7,
    w_meta: float = 0.3,
) -> dict[str, float]:
    """Combined score: weighted bit Tanimoto + meta cosine."""
    s_bits = tanimoto_bits(a, b)
    s_meta = cosine_meta(a, b)
    return {
        "tanimoto_bits": s_bits,
        "cosine_meta": s_meta,
        "combined": w_bits * s_bits + w_meta * s_meta,
    }


def describe_layout() -> str:
    return (
        f"complex fingerprint dim={TOTAL_DIM}\n"
        f"  bits: Morgan({MORGAN_BITS}) + MACCS({MACCS_BITS}) + RDKit({RDKIT_BITS}) = {BIT_DIM}\n"
        f"  meta: metal_onehot({len(METAL_SYMBOLS)}) + charge_onehot({CHARGE_DIM}) + d_onehot({D_ELECTRON_DIM}) = {META_DIM}"
    )


if __name__ == "__main__":
    base = "F[C-](F)(F)->[METAL](<-[O-][n+]1ccccc1)(<-[C-](F)(F)F)<-[C-](F)(F)F"
    cases = {
        "Au+2": "[Au+2]",
        "Pd+2": "[Pd+2]",
        "Cu+2": "[Cu+2]",
        "Fe+2": "[Fe+2]",
        "Fe+3": "[Fe+3]",
    }
    fps = {
        name: build_complex_fingerprint(base.replace("[METAL]", token))
        for name, token in cases.items()
    }

    print(describe_layout(), "\n")
    for name, fp in fps.items():
        print(
            f"{name}: metal={fp.metal_symbol} q={fp.formal_charge:+d} "
            f"d={fp.d_electrons} | on-bits={int(fp.bit_vector.sum())}"
        )

    print("\n=== Pairwise combined similarity (w_bits=0.7, w_meta=0.3) ===")
    names = list(cases)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s = similarity(fps[a], fps[b])
            print(
                f"  {a} vs {b}: bits={s['tanimoto_bits']:.3f} "
                f"meta={s['cosine_meta']:.3f} combined={s['combined']:.3f}"
            )
