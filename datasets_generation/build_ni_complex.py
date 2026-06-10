"""
One case directory: use MetalloGen to assemble a five-coordinate square-pyramidal Ni complex
from a bidentate ligand and three organic substrate fragments; write complex_Ni.xyz in the same folder.

The case directory must contain:
  - Ligand*_N*.xyz (filename must contain N{i}_N{j} for the two donor nitrogens);
  - R1_stay*.xyz, R2_stay*.xyz, R2_leave*.xyz (optional site index in the filename).

---
MetalloGen workflow sketch:

1) Build chem.Molecule from XYZ:
   - Read XYZ -> chem.Atom(sym) with x,y,z -> mol.atom_list
   - adj = process.get_adj_matrix_from_distance(mol, coeff=1.10)
   - chg_list, bo = process.get_chg_and_bo(mol, chg=chg)
   - mol.adj_matrix = adj, mol.bo_matrix = bo, mol.chg, mol.multiplicity

2) Build Ligand per fragment and set binding_infos:
   - lig = metallogen_ligand.Ligand(mol, [])
   - Per donor: lig.binding_infos.append(([local_atom_index], site))
   - site is 1..steric_number (geometry site index)

3) Geometry and MetalComplex:
   - geometry = metallogen_om.Geometry(geometry_name)  # e.g. "5_square_pyramidal"
   - center_atom = chem.Atom(metal_symbol)
   - metal_complex = metallogen_om.MetalComplex(
         geometry_name, center_atom, [lig1, lig2, ...], charge, multiplicity)
   - metal_complex.metal_index = 0

4) TMCGenerator for conformers (xtb relaxation):
   - calc = metallogen_gaussian.Gaussian(); calc.switch_to_xtb_gaussian()
   - calc.change_working_directory(working_directory)
   - generator = metallogen_run.TMCGenerator(calculator=calc, scale=1.0, align=True)
   - ace_mols = generator.sample_conformer(metal_complex)

5) Write XYZ:
   - coords = mol.get_coordinate_list(); elems = [a.get_element() for a in mol.atom_list]

---
Common failure modes (energy=1e6, Written 0):

1) Embed "Atoms are too close":
   - RDKit initial geometry has ligand/substrate atoms too close; MetalloGen embed
     ratio_criteria / atom_d_criteria reject both options; only a poor fallback remains (score -50000).

2) FF clean fails:
   - Poor starting geometry; force-field clean cannot separate atoms; proceeds to "Further cleaning with QC".

3) QC "Calculation was not successful":
   - clean_geometry calls calculator.relax_geometry() then get_energy().
   - If energy is None, QC failed. Causes may include:
     a) Gaussian (g09/g16) missing or not on PATH (pipeline uses Gaussian + external=xtb via xtbbin).
     b) xtbbin xtb-gaussian wrapper missing or failing (set xtbbin=/path/to/xtb-gaussian).
     c) Bad geometry: no convergence or crash.
     d) .log not parseable by cclib (e.g. no SCF energy), get_energy returns None.

4) Outcome:
   - sample_conformer keeps only conformers with energy < 1e6; all failing returns [].

"""

import os
import re
import sys
import glob
import argparse
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from contextlib import redirect_stdout, redirect_stderr

# xtb-gaussian executable; set each run for MetalloGen Gaussian external interface.
if "xtbbin" not in os.environ:
    os.environ["xtbbin"] = "/path/to/xtb-gaussian/xtb-gaussian"

class Tee:
    """Mirror write() to multiple streams (e.g. terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            if hasattr(s, "flush"):
                s.flush()

    def flush(self):
        for s in self.streams:
            if hasattr(s, "flush"):
                s.flush()

# MetalloGen lives at <repo>/MetalloGen/ (sibling of datasets_generation); add repo root to sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_MG_PKG = os.path.join(_REPO_ROOT, "MetalloGen")
if os.path.isfile(os.path.join(_MG_PKG, "__init__.py")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MetalloGen import chem, process
from MetalloGen import embed as metallogen_embed
from MetalloGen import ligand as metallogen_ligand
from MetalloGen import om as metallogen_om
from MetalloGen import run as metallogen_run
from MetalloGen.Calculator import gaussian as metallogen_gaussian


def load_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    """Read XYZ file; return (element symbols, Nx3 coordinates)."""
    # XYZ layout:
    #   line 1: atom count
    #   line 2: comment (may contain energy/gnorm/xtb text, or be empty)
    #   line 3+: element + x y z (optional extra columns)
    with open(path, encoding="utf-8", errors="replace") as f:
        raw_lines = [l.rstrip("\n") for l in f]
    if not raw_lines:
        return [], np.zeros((0, 3))

    # Skip leading blank lines
    start = 0
    while start < len(raw_lines) and not raw_lines[start].strip():
        start += 1
    if start >= len(raw_lines):
        return [], np.zeros((0, 3))

    n = int(raw_lines[start].strip().split()[0])
    coord_start = start + 2  # skip comment line after count
    if len(raw_lines) < coord_start + n:
        return [], np.zeros((0, 3))
    symbols = []
    coords = []
    for i in range(coord_start, coord_start + n):
        parts = raw_lines[i].strip().split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords)


def _build_chem_molecule_from_xyz(xyz_path: str, chg: int = 0) -> chem.Molecule:
    """Build MetalloGen chem.Molecule from XYZ (distance-based adjacency, then charges/bond orders)."""
    symbols, coords = load_xyz(xyz_path)
    if not symbols:
        raise ValueError(f"Empty or invalid XYZ: {xyz_path}")
    mol = chem.Molecule()
    atom_list = []
    for sym, xyz in zip(symbols, coords):
        a = chem.Atom(sym)
        a.x, a.y, a.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        atom_list.append(a)
    mol.atom_list = atom_list
    adj = process.get_adj_matrix_from_distance(mol, coeff=1.10)
    mol.adj_matrix = adj
    chg_list, bo = process.get_chg_and_bo(mol, chg=chg)
    mol.bo_matrix = bo
    mol.set_atom_feature(np.array(chg_list), "chg")
    mol.chg = chg
    mol.multiplicity = 1
    return mol


def parse_ligand_donors_from_filename(basename: str) -> Optional[Tuple[int, int]]:
    """Parse two donor N 0-based indices from filename, e.g. Ligand_N13_N4.xyz -> (13, 4)."""
    m = re.search(r"N(\d+)_N(\d+)", basename, re.I)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_site_from_filename(basename: str) -> Optional[int]:
    """Parse site index from filename, e.g. R1_stay_site3.xyz -> 3."""
    m = re.search(r"site(\d+)", basename, re.I)
    if not m:
        return None
    return int(m.group(1))


def resolve_substrate_paths_from_dir(folder: str) -> Optional[Dict[str, Tuple[str, int]]]:
    """Resolve R1_stay*.xyz, R2_stay*.xyz, R2_leave*.xyz paths and site indices from one directory."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return None
    out: Dict[str, Tuple[str, int]] = {}
    for role, pattern in [
        ("R1_stay", "R1_stay*.xyz"),
        ("R2_stay", "R2_stay*.xyz"),
        ("R2_leave", "R2_leave*.xyz"),
    ]:
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not files:
            return None
        path = files[0]
        site = parse_site_from_filename(os.path.basename(path))
        if site is None:
            site = 0
        out[role] = (path, site)
    return out


def resolve_ligand_path_from_dir(folder: str) -> Optional[Tuple[str, int, int]]:
    """Pick first Ligand*_N*.xyz in folder and parse donors (filename must contain N{i}_N{j})."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return None
    candidates = sorted(glob.glob(os.path.join(folder, "Ligand*_N*.xyz")))
    if not candidates:
        return None
    path = candidates[0]
    donors = parse_ligand_donors_from_filename(os.path.basename(path))
    if donors is None:
        return None
    return path, donors[0], donors[1]


def _total_electrons(
    lig_mol: chem.Molecule,
    r1_mol: chem.Molecule,
    r2s_mol: chem.Molecule,
    r2l_mol: chem.Molecule,
    metal_symbol: str = "Ni",
) -> int:
    """Total electron count for four fragments plus the metal."""
    total = 0
    for mol in (lig_mol, r1_mol, r2s_mol, r2l_mol):
        for a in mol.atom_list:
            total += a.get_atomic_number()
    total += chem.Atom(metal_symbol).get_atomic_number()
    return total


def _auto_multiplicity(
    total_electrons: int, charge: int, requested: int
) -> int:
    """
    Adjust spin multiplicity from electron count and charge.
    Odd electron count cannot be singlet; use at least 2 to avoid Gaussian
    "The combination of multiplicity 1 and N electrons is impossible".
    """
    n_electrons = total_electrons - charge
    if n_electrons % 2 == 0:
        return requested
    return max(requested, 2)


def build_complex_from_parts(
    ligand_path: str,
    n1: int,
    n2: int,
    r1_path: str,
    r1_site: int,
    r2s_path: str,
    r2s_site: int,
    r2l_path: str,
    r2l_site: int,
    working_directory: str,
    complex_charge: int = 0,
    multiplicity: int = 1,
    calculator: str = "xtb_gaussian",
    fallback_embed: bool = False,
) -> Optional[Tuple[List[chem.Molecule], int, int, bool]]:
    """
    Run MetalloGen square-pyramidal Ni assembly from ligand + three substrate XYZ paths.
    Returns (ace_mols, charge_used, multiplicity_used, is_embed_only) or None on failure.
    """
    try:
        lig_mol = _build_chem_molecule_from_xyz(ligand_path, chg=0)
        r1_mol = _build_chem_molecule_from_xyz(r1_path, chg=0)
        r2s_mol = _build_chem_molecule_from_xyz(r2s_path, chg=0)
        r2l_mol = _build_chem_molecule_from_xyz(r2l_path, chg=0)
    except Exception:
        return None
    n_lig, n_r1, n_r2s, n_r2l = len(lig_mol.atom_list), len(r1_mol.atom_list), len(r2s_mol.atom_list), len(r2l_mol.atom_list)
    if n1 >= n_lig or n2 >= n_lig or r1_site >= n_r1 or r2s_site >= n_r2s or r2l_site >= n_r2l:
        return None

    geometry_name = "5_square_pyramidal"
    lig_lig = metallogen_ligand.Ligand(lig_mol, [])
    lig_lig.binding_infos.append(([n1], 1))
    lig_lig.binding_infos.append(([n2], 3))
    lig_r1 = metallogen_ligand.Ligand(r1_mol, [])
    lig_r1.binding_infos.append(([r1_site], 5))
    lig_r2s = metallogen_ligand.Ligand(r2s_mol, [])
    lig_r2s.binding_infos.append(([r2s_site], 2))
    lig_r2l = metallogen_ligand.Ligand(r2l_mol, [])
    lig_r2l.binding_infos.append(([r2l_site], 4))

    total_electrons = _total_electrons(lig_mol, r1_mol, r2s_mol, r2l_mol, metal_symbol="Ni")
    multiplicity = _auto_multiplicity(total_electrons, complex_charge, multiplicity)

    center_atom = chem.Atom("Ni")
    metal_complex = metallogen_om.MetalComplex(
        geometry_name,
        center_atom,
        [lig_lig, lig_r1, lig_r2s, lig_r2l],
        complex_charge,
        multiplicity,
    )
    metal_complex.metal_index = 0

    if calculator == "xtb_gaussian":
        calc = metallogen_gaussian.Gaussian()
        ok = calc.switch_to_xtb_gaussian()
        if not ok:
            return None
    else:
        return None
    os.makedirs(working_directory, exist_ok=True)
    calc.change_working_directory(working_directory)

    generator = metallogen_run.TMCGenerator(
        calculator=calc,
        scale=1.0,
        align=True,
        always_qc=False,
    )
    ace_mols = generator.sample_conformer(metal_complex)
    if ace_mols:
        return (ace_mols, complex_charge, multiplicity, False)
    if fallback_embed:
        for option in (0, 1):
            positions = metallogen_embed.get_embedding(
                metal_complex, scale=1.0, option=option, align=True, use_random=True
            )
            if positions is not None:
                metal_complex.set_position(positions)
                fallback_mol = metal_complex.get_molecule()
                return ([fallback_mol], complex_charge, multiplicity, True)
    return None


def write_xyz_from_ace_mol(ace_mol: chem.Molecule, path: str, charge: int = 0, multiplicity: int = 1) -> None:
    """Write XYZ from MetalloGen ace_mol."""
    coords = ace_mol.get_coordinate_list()
    elems = [a.get_element() for a in ace_mol.atom_list]
    n = len(elems)
    energy = getattr(ace_mol, "energy", 0.0)
    with open(path, "w") as f:
        f.write(f"{n}\n{charge}\t{multiplicity}\t{energy}\n")
        for el, (x, y, z) in zip(elems, coords):
            f.write(f"{el}  {x:.6f}  {y:.6f}  {z:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a five-coordinate square-pyramidal Ni(II) complex from a bidentate ligand "
            "and three substrate fragments; writes complex_Ni.xyz."
        ),
    )
    parser.add_argument(
        "case_dir",
        metavar="CASE_DIR",
        help=(
            "Case directory containing Ligand*_N*.xyz (bidentate ligand) and "
            "R1_stay/R2_stay/R2_leave substrate XYZ files; output written here."
        ),
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        default=None,
        help="MetalloGen scratch directory (default: <CASE_DIR>/_metallogen_wd).",
    )
    parser.add_argument("--charge", type=int, default=0, help="Overall complex charge.")
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (raised to at least 2 for odd electron count).",
    )
    parser.add_argument(
        "--fallback-embed",
        action="store_true",
        help="If QC yields no geometry, try embed-only; writes complex_Ni_embed_only.xyz.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress MetalloGen stdout/stderr.")
    parser.add_argument("--log", default=None, metavar="FILE", help="Also tee stdout/stderr to this file.")
    args = parser.parse_args()

    log_file = None
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    if args.log:
        log_file = open(args.log, "w", encoding="utf-8")
        sys.stdout = Tee(orig_stdout, log_file)
        sys.stderr = Tee(orig_stderr, log_file)
    try:
        assemble_ni_complex(args)
    finally:
        if args.log and log_file is not None:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


def assemble_ni_complex(args: argparse.Namespace) -> None:
    """Assemble Ni complex from ligand and substrates in case_dir; write complex_Ni.xyz there."""
    case_dir = os.path.abspath(args.case_dir)
    if not os.path.isdir(case_dir):
        raise SystemExit(f"Not a directory: {case_dir}")

    subs = resolve_substrate_paths_from_dir(case_dir)
    if subs is None:
        raise SystemExit(
            f"Case directory must contain R1_stay*.xyz, R2_stay*.xyz, and R2_leave*.xyz: {case_dir}"
        )
    ligand_res = resolve_ligand_path_from_dir(case_dir)
    if ligand_res is None:
        raise SystemExit(
            f"Missing ligand file Ligand*_N*.xyz (filename must contain N<number>_N<number>): {case_dir}"
        )

    ligand_path, n1, n2 = ligand_res
    r1_path, r1_site = subs["R1_stay"]
    r2s_path, r2s_site = subs["R2_stay"]
    r2l_path, r2l_site = subs["R2_leave"]

    if args.work_dir:
        wd = os.path.join(os.path.abspath(args.work_dir), "metallogen_run")
    else:
        wd = os.path.join(case_dir, "_metallogen_wd")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        from rdkit.rdBase import RandomSeed as RDKitRandomSeed  # type: ignore[import-untyped]
        RDKitRandomSeed(seed)
    except Exception:
        pass
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    def _run():
        return build_complex_from_parts(
            ligand_path,
            n1,
            n2,
            r1_path,
            r1_site,
            r2s_path,
            r2s_site,
            r2l_path,
            r2l_site,
            working_directory=wd,
            complex_charge=args.charge,
            multiplicity=args.multiplicity,
            calculator="xtb_gaussian",
            fallback_embed=args.fallback_embed,
        )

    try:
        if args.quiet:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    result = _run()
        else:
            result = _run()
    except Exception as e:
        raise SystemExit(f"MetalloGen error: {e}") from e

    if result is None:
        raise SystemExit(
            "No valid geometry. Check xtbbin / Gaussian, or try --fallback-embed for embed-only."
        )
    ace_mols, charge_used, multiplicity_used, is_embed_only = result
    base = "complex_Ni_embed_only" if is_embed_only else "complex_Ni"
    out_xyz = os.path.join(case_dir, f"{base}.xyz")
    write_xyz_from_ace_mol(
        ace_mols[0],
        out_xyz,
        charge=charge_used,
        multiplicity=multiplicity_used,
    )
    print(f"Wrote: {out_xyz}")


if __name__ == "__main__":
    main()
