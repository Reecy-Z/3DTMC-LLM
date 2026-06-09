"""
Supported OOD split definitions for Stage 3 downstream tasks.

  - Vaska: 26 leave-one-ligand-out folds (``OOD.Vaska.ligand_split.LIGANDS``)
  - NiComplex: 3 ligand-scaffold holdouts — Pybox, Biox, Biim
  - Property: user-provided CSV with ``CSD code`` + ``split`` (or train.csv / test.csv dir)
"""
from OOD.NiComplex.nicomplex_split import EXPERIMENT_NAMES as NICOMPLEX_SCAFFOLD_OOD_EXPERIMENTS
from OOD.Vaska.ligand_split import LIGANDS as VASKA_LIGAND_OOD_LIGANDS

__all__ = [
    "NICOMPLEX_SCAFFOLD_OOD_EXPERIMENTS",
    "VASKA_LIGAND_OOD_LIGANDS",
]
