"""
Central registry for tasks, instructions, and dataset wiring across Stage 1/2/3.

Instructions and task metadata for Stage 1/2/3. ``task_datasets.py`` consumes this
module for prompt assembly; Stage*.py migration: docs/STAGE3_MERGE_PLAN.md

Usage (target):
    from task_registry import get_task, resolve_user_content, list_tasks

    task = get_task("vaska_barrier")
    text = resolve_user_content(task, mode="single_token", record={"smiles": "..."})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from utils import format_instruction_field

# ---------------------------------------------------------------------------
# Modes (structure / training recipe axis; same vocabulary train + infer)
# ---------------------------------------------------------------------------

Mode = Literal["single_token", "multi_token", "3d_only", "freeze_3d", "random_3d"]

MODES_STAGE1: tuple[Mode, ...] = ("single_token", "multi_token", "3d_only")
MODES_STAGE2: tuple[Mode, ...] = (
    "single_token",
    "multi_token",
    "3d_only",
    "freeze_3d",
    "random_3d",
)
MODES_STAGE3: tuple[Mode, ...] = MODES_STAGE2

# Default mode per stage when --mode is omitted
DEFAULT_MODE: dict[int, Mode] = {1: "single_token", 2: "single_token", 3: "single_token"}


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------

SplitStrategy = Literal[
    "fixed_lmdb",           # explicit train_lmdb + val_lmdb (TmQM property / description)
    "random_80_10_10",      # single --lmdb, shuffle with split_seed
    "nicomplex_ood",        # OOD/NiComplex/nicomplex_split.py
    "vaska_loobo_b_group",  # OOD/Vaska/b_group_split.py
    "vaska_ligand_loobo",   # OOD/Vaska/ligand_split.py
]

TaskKind = Literal["description", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    """One train/infer task: instruction(s), target field, data expectations."""

    name: str
    stages: tuple[int, ...]
    kind: TaskKind

    # LMDB / JSON fields
    target_key: str
    response_key_stage1: Optional[str] = None  # description task, Stage1 uses raw description
    response_key_stage2: Optional[str] = None  # polished_description

    # Instructions (mode selects which template)
    instruction_smiles: Optional[str] = None
    instruction_3d_only: Optional[str] = None
    instruction_description: Optional[str] = None  # prepend polished_description (homo_lumo)
    instruction_template: Optional[str] = None  # str.format placeholders, e.g. {smiles} {temp}
    instruction_fields: tuple[str, ...] = ()  # keys passed to .format()

    # Prompt behaviour
    include_smiles_default: bool = True
    requires_smiles_in_lmdb: bool = True  # filter records missing smiles
    use_polished_description_option: bool = False

    # Data / splits
    split_strategy: SplitStrategy = "fixed_lmdb"
    default_train_lmdb: Optional[str | Sequence[str]] = None
    default_val_lmdb: Optional[str] = None
    default_lmdb: Optional[str | Sequence[str]] = None

    # Metadata
    unit: Optional[str] = None
    output_dir_suffix: Optional[str] = None
    wandb_project_prefix: Optional[str] = None

    def supports_stage(self, stage: int) -> bool:
        return stage in self.stages

    def response_key_for_stage(self, stage: int) -> str:
        if self.kind == "description":
            if stage == 1:
                return self.response_key_stage1 or "description"
            return self.response_key_stage2 or "polished_description"
        return self.target_key


# ---------------------------------------------------------------------------
# Instruction strings (single source of truth)
# ---------------------------------------------------------------------------

_DESC_SMILES = "Give me a description of this transition metal complex:"
_DESC_3D_ONLY = "Give a description of this transition metal complex:"

_PROP_DIPOLE_SMILES = (
    "What is the dipole moment (in Debye) of this transition metal complex? "
    "Given the SMILES and structure, respond with the numerical value only:"
)
_PROP_DIPOLE_3D = (
    "What is the dipole moment (in Debye) of this transition metal complex? "
    "Given the 3D structure only, respond with the numerical value only:"
)
_PROP_POLAR_SMILES = (
    "What is the polarisability (in Bohr^3) of this transition metal complex? "
    "Given the SMILES and structure, respond with the numerical value only:"
)
_PROP_POLAR_3D = (
    "What is the polarisability (in Bohr^3) of this transition metal complex? "
    "Given the 3D structure only, respond with the numerical value only:"
)
_PROP_GAP_SMILES = (
    "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? "
    "Given the SMILES and structure, respond with the numerical value only:"
)
_PROP_GAP_DESC = (
    "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? "
    "Given the description, SMILES and structure, respond with the numerical value only:"
)
_PROP_GAP_3D = (
    "What is the HOMO-LUMO gap (in Ha) of this transition metal complex? "
    "Given the 3D structure only, respond with the numerical value only:"
)

_VASKA_BARRIER = (
    "What is the dihydrogen activation energy barrier (in kcal/mol) of this Vaska's complex? "
    "Given the SMILES and structure, respond with the numerical value only:"
)
_NICOMPLEX_DDG = (
    "The key Ni(III) intermediate complex, bearing chiral ligands and carbon groups, "
    "determines the enantioselectivity (\u2206\u2206G) of nickel-catalyzed cross-coupling reactions. "
    "What is the \u2206\u2206G (in kcal/mol) for this reaction? "
    "Given the Ni(III) intermediate {smiles} and {temp} K, respond with the numerical value only:"
)


# ---------------------------------------------------------------------------
# Task table
# ---------------------------------------------------------------------------

TASKS: dict[str, TaskSpec] = {
    # ---- Stage 1 / 2: description -----------------------------------------
    "description": TaskSpec(
        name="description",
        stages=(1, 2),
        kind="description",
        target_key="polished_description",
        response_key_stage1="description",
        response_key_stage2="polished_description",
        instruction_smiles=_DESC_SMILES,
        instruction_3d_only=_DESC_3D_ONLY,
        include_smiles_default=True,
        split_strategy="fixed_lmdb",
        wandb_project_prefix="Stage",
    ),
    # ---- Stage 3: TmQM properties -----------------------------------------
    "dipole_moment": TaskSpec(
        name="dipole_moment",
        stages=(3,),
        kind="regression",
        target_key="dipole_moment",
        instruction_smiles=_PROP_DIPOLE_SMILES,
        instruction_3d_only=_PROP_DIPOLE_3D,
        unit="Debye",
        output_dir_suffix="dipole_moment",
        wandb_project_prefix="Stage3_Property",
        split_strategy="fixed_lmdb",
    ),
    "polarisability": TaskSpec(
        name="polarisability",
        stages=(3,),
        kind="regression",
        target_key="polarisability",
        instruction_smiles=_PROP_POLAR_SMILES,
        instruction_3d_only=_PROP_POLAR_3D,
        unit="Bohr^3",
        output_dir_suffix="polarisability",
        wandb_project_prefix="Stage3_Property",
        split_strategy="fixed_lmdb",
    ),
    "homo_lumo_gap": TaskSpec(
        name="homo_lumo_gap",
        stages=(3,),
        kind="regression",
        target_key="homo_lumo_gap",
        instruction_smiles=_PROP_GAP_SMILES,
        instruction_3d_only=_PROP_GAP_3D,
        instruction_description=_PROP_GAP_DESC,
        use_polished_description_option=True,
        unit="Ha",
        output_dir_suffix="homo_lumo_gap",
        wandb_project_prefix="Stage3_Property",
        split_strategy="fixed_lmdb",
    ),
    # ---- Stage 3: downstream (same stack, different instruction) ----------
    "vaska_barrier": TaskSpec(
        name="vaska_barrier",
        stages=(3,),
        kind="regression",
        target_key="barrier",
        instruction_smiles=_VASKA_BARRIER,
        instruction_3d_only=None,  # not supported for Vaska mainline
        requires_smiles_in_lmdb=True,
        unit="kcal/mol",
        output_dir_suffix="vaska_barrier",
        wandb_project_prefix="Stage3_Vaska",
        split_strategy="random_80_10_10",
        default_lmdb="/path/to/vaskas-space/data.lmdb",
    ),
    "nicomplex_ddg": TaskSpec(
        name="nicomplex_ddg",
        stages=(3,),
        kind="regression",
        target_key="ddG",
        instruction_template=_NICOMPLEX_DDG,
        instruction_fields=("smiles", "temp"),
        include_smiles_default=True,  # smiles embedded in template, not appended
        requires_smiles_in_lmdb=False,
        unit="kcal/mol",
        output_dir_suffix="nicomplex_ddg",
        wandb_project_prefix="Stage3_NiComplex",
        split_strategy="random_80_10_10",
        default_lmdb=["/path/to/NiComplex/data.lmdb"],
    ),
}

# Backward-compatible aliases (old --property / script names)
TASK_ALIASES: dict[str, str] = {
    "property": "dipole_moment",  # only when used with explicit --property; prefer task name
    "vaska": "vaska_barrier",
    "nicomplex": "nicomplex_ddg",
}

STAGE3_TASKS: tuple[str, ...] = tuple(
    name for name, spec in TASKS.items() if 3 in spec.stages and spec.kind == "regression"
)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def normalize_task_name(name: str) -> str:
    name = name.strip()
    return TASK_ALIASES.get(name, name)


def get_task(name: str) -> TaskSpec:
    name = normalize_task_name(name)
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]


def list_tasks(*, stage: Optional[int] = None, kind: Optional[TaskKind] = None) -> list[str]:
    out = []
    for name, spec in TASKS.items():
        if stage is not None and not spec.supports_stage(stage):
            continue
        if kind is not None and spec.kind != kind:
            continue
        out.append(name)
    return sorted(out)


def mode_includes_smiles(mode: Mode) -> bool:
    return mode not in ("3d_only",)


def resolve_instruction(
    task: TaskSpec | str,
    *,
    mode: Mode = "single_token",
    use_polished_description: bool = False,
    instruction_override: Optional[str] = None,
) -> str:
    """Return the instruction template for this task + mode (before record formatting)."""
    if instruction_override is not None:
        return instruction_override

    if isinstance(task, str):
        task = get_task(task)

    if not mode_includes_smiles(mode):
        if task.instruction_3d_only:
            return task.instruction_3d_only
        if task.instruction_template and not task.instruction_fields:
            return task.instruction_template
        raise ValueError(f"Task {task.name!r} has no instruction_3d_only for mode={mode!r}")

    if use_polished_description and task.instruction_description:
        return task.instruction_description
    if task.instruction_template:
        return task.instruction_template
    if task.instruction_smiles:
        return task.instruction_smiles
    raise ValueError(f"Task {task.name!r} has no instruction for mode={mode!r}")


def resolve_user_content(
    task: TaskSpec | str,
    record: Mapping[str, Any],
    *,
    mode: Mode = "single_token",
    use_polished_description: bool = False,
    instruction_override: Optional[str] = None,
) -> str:
    """
    Build the user-side chat text: instruction (+ optional SMILES / description / format fields).

    Target sample_builder entry point once datasets are migrated.
    """
    if isinstance(task, str):
        task = get_task(task)

    instruction = resolve_instruction(
        task,
        mode=mode,
        use_polished_description=use_polished_description,
        instruction_override=instruction_override,
    )

    # Templated instructions (NiComplex: {smiles}, {temp})
    if task.instruction_template or ("{" in instruction and "}" in instruction):
        fmt: dict[str, str] = {}
        for key in task.instruction_fields:
            fmt[key] = format_instruction_field(record.get(key, ""))
        if fmt:
            return instruction.format(**fmt)
        return instruction

    # 3d_only: instruction only
    if not mode_includes_smiles(mode):
        return instruction.strip()

    smiles = format_instruction_field(record.get("smiles", ""))

    # homo_lumo: optional polished_description prefix
    if use_polished_description and task.use_polished_description_option:
        desc = record.get("polished_description") or record.get("description")
        if desc is not None:
            desc = str(desc).strip()
        if desc:
            return f"{desc}\n{instruction} {smiles}".strip()

    if task.instruction_template:
        # template already consumed smiles via format
        return instruction.strip()

    return f"{instruction} {smiles}".strip()


def format_target_text(task: TaskSpec | str, record: Mapping[str, Any], *, stage: int = 3) -> str:
    """Generation target string (description or numeric)."""
    if isinstance(task, str):
        task = get_task(task)
    key = task.response_key_for_stage(stage)
    value = record[key]
    if task.kind == "regression":
        return f"{float(value):.6f}"
    text = value
    if text is None:
        raise ValueError(f"Missing {key!r} in record")
    return str(text).strip()


def default_output_dir(task: TaskSpec | str, mode: Mode, *, base: Optional[str] = None) -> str:
    """Suggested checkpoint directory (mirrors current Property.py / train_defaults patterns)."""
    if isinstance(task, str):
        task = get_task(task)
    suffix = task.output_dir_suffix or task.name
    if base:
        return base
    if mode in ("freeze_3d", "random_3d", "multi_token", "3d_only"):
        return f"/path/to/Stage3_{suffix}_{mode}_ckpt"
    return f"/path/to/Stage3_{suffix}_ckpt"


def wandb_project(task: TaskSpec | str, mode: Mode) -> str:
    if isinstance(task, str):
        task = get_task(task)
    prefix = task.wandb_project_prefix or "Stage3"
    suffix = task.output_dir_suffix or task.name
    if mode == "single_token":
        return f"{prefix}_{suffix}"
    return f"{prefix}_{suffix}_{mode}"


# ---------------------------------------------------------------------------
# Bridge: legacy PROPERTY_CONFIG shape (for gradual migration)
# ---------------------------------------------------------------------------


def property_config_dict() -> dict[str, dict[str, str]]:
    """Drop-in replacement shape for task_datasets.PROPERTY_CONFIG."""
    out = {}
    for name in ("dipole_moment", "polarisability", "homo_lumo_gap"):
        t = TASKS[name]
        entry: dict[str, str] = {
            "key": t.target_key,
            "unit": t.unit or "",
            "instruction_smiles": t.instruction_smiles or "",
            "output_dir_suffix": t.output_dir_suffix or name,
        }
        if t.instruction_description:
            entry["instruction_description"] = t.instruction_description
        out[name] = entry
    return out


def instruction_3d_only_dict() -> dict[str, str]:
    """Drop-in for task_datasets.INSTRUCTION_3D_ONLY."""
    return {
        name: TASKS[name].instruction_3d_only or ""
        for name in ("dipole_moment", "polarisability", "homo_lumo_gap")
    }


# Public constants re-exported for Stage1/2 migration
INSTRUCTION_DESCRIPTION = _DESC_SMILES
INSTRUCTION_DESCRIPTION_3D_ONLY = _DESC_3D_ONLY
VASKA_INSTRUCTION = _VASKA_BARRIER
NI_INSTRUCTION = _NICOMPLEX_DDG

__all__ = [
    "DEFAULT_MODE",
    "INSTRUCTION_DESCRIPTION",
    "INSTRUCTION_DESCRIPTION_3D_ONLY",
    "MODES_STAGE1",
    "MODES_STAGE2",
    "MODES_STAGE3",
    "NI_INSTRUCTION",
    "STAGE3_TASKS",
    "TASKS",
    "TASK_ALIASES",
    "Mode",
    "SplitStrategy",
    "TaskKind",
    "TaskSpec",
    "VASKA_INSTRUCTION",
    "default_output_dir",
    "format_target_text",
    "get_task",
    "instruction_3d_only_dict",
    "list_tasks",
    "mode_includes_smiles",
    "normalize_task_name",
    "property_config_dict",
    "resolve_instruction",
    "resolve_user_content",
    "wandb_project",
]
