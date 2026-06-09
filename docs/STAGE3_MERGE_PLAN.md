# Stage3 合并方案（草案）

目标：把 `Stage3/Property.py`、`Stage3/NiComplex.py`、`Stage3/Vaska_Complex.py` 合并为 **一个** `Stage3.py`（与 `Stage1.py` / `Stage2.py` 同级），任务差异由 `task_registry.py` 的 `--task` 驱动；`--ablation` 统一改名为 `--mode`。

配套：`task_registry.py`（已起草，尚未接入训练脚本）。

---

## 1. 合并后的入口

```
3DTMC-LLM/
  Stage1.py          # 不变（后续接 task_registry）
  Stage2.py          # 不变（后续接 task_registry）
  Stage3.py          # NEW：合并 Property + NiComplex + Vaska
  task_registry.py   # NEW：instruction / task 元数据
  task_datasets.py   # 保留：Dataset 类；逐步改为调用 resolve_user_content()
  multimodal_LLM.py
  utils.py
  inference.py       # 扩展：Stage3 全 task + 全 mode
```

`Stage3/` 目录在合并后 **不再放主训练脚本**；OOD 的 split 工具可暂留 `OOD/` 下，由 `Stage3.py --split` 调用。

---

## 2. 统一 CLI：`Stage3.py`

### 2.1 必选 / 核心参数

| 参数 | 说明 | 默认来源 |
|------|------|----------|
| `--task` | 任务名 | `dipole_moment`（或保留 `--property` 作 alias） |
| `--mode` | 结构/训练模式 | `single_token` |

**`--task` 可选值（Stage3）**

| task | 原脚本 | target 字段 |
|------|--------|-------------|
| `dipole_moment` | Property.py | `dipole_moment` |
| `polarisability` | Property.py | `polarisability` |
| `homo_lumo_gap` | Property.py | `homo_lumo_gap` |
| `vaska_barrier` | Vaska_Complex.py | `barrier` |
| `nicomplex_ddg` | NiComplex.py | `ddG` |

**`--mode` 可选值**（取代原 `--ablation`）

| mode | 原 `--ablation` | 行为摘要 |
|------|-----------------|----------|
| `single_token` | `property` | 默认：instruction+SMILES+3D，训 encoder+proj+LoRA |
| `freeze_3d` | `freeze_3d` | 冻 3D+projection，只训 LoRA |
| `random_3d` | `random_3d` | 随机 3D slot，只训 LoRA |
| `multi_token` | `multi_token` | k-query projection |
| `3d_only` | `3d_only` | prompt 无 SMILES |

向后兼容（建议保留 1 个版本周期）：

```bash
# 旧
deepspeed Stage3/Property.py --ablation freeze_3d --property dipole_moment
# 新
deepspeed Stage3.py --task dipole_moment --mode freeze_3d
# 兼容层（可选）
--ablation freeze_3d  →  映射到 --mode freeze_3d
--property dipole_moment  →  映射到 --task dipole_moment
```

### 2.2 模型与 checkpoint

| 参数 | 保留 | 备注 |
|------|------|------|
| `--model_name` | ✅ | |
| `--3D_encoder_dict` | ✅ | dest `three_d_encoder_dict` |
| `--Stage2_ckpt` | ✅ | dest `stage2_ckpt` |
| `--3D_encoder_ckpt` | ✅ | 可选覆盖 |
| `--projection_init` | ✅ | `pretrained` / `from_scratch` |
| `--lora_r` / `--lora_alpha` / `--lora_target` | ✅ | |
| `--lora_init` | ✅ | multi_token / 3d_only ablation |
| `--train_3d_encoder` | ✅ | BooleanOptionalAction |
| `--train_projection` | ✅ | |
| `--train_lora` | ✅ | |

### 2.3 数据与划分

按 `task_registry.TaskSpec.split_strategy` 分支：

| split_strategy | 适用 task | CLI |
|----------------|-----------|-----|
| `fixed_lmdb` | dipole / polarisability / homo_lumo | `--train_lmdb`, `--val_lmdb` |
| `random_80_10_10` | vaska_barrier, nicomplex_ddg | `--lmdb`（可 append 多次）, `--split_seed` |
| `vaska_loobo_b_group` | vaska OOD | `--lmdb`, `--holdout_b_group`, `--run_all_loops` |
| `nicomplex_ood` | NiComplex OOD | `--lmdb`, OOD split 参数（见 OOD 脚本） |

| 参数 | 保留 | 备注 |
|------|------|------|
| `--train_lmdb` | ✅ | TmQM property |
| `--val_lmdb` | ✅ | TmQM property |
| `--lmdb` | ✅ | Vaska / NiComplex（`action=append`） |
| `--split_seed` | ✅ | 80/10/10 → `output_dir/seed_<seed>/` |
| `--split` | 🆕 | `fixed` / `random_80_10_10` / `loobo_b_group` / … |
| `--max_samples` | ✅ | smoke test |
| `--max_train_samples` | ✅ | NiComplex 专用，可泛化为 `--max_train_samples` |
| `--max_eval_samples` | ✅ | 同上 |
| `--use_polished_description` | ✅ | homo_lumo_gap |

### 2.4 训练超参

| 参数 | 保留 | 默认 |
|------|------|------|
| `--output_dir` | ✅ | task_registry 或 train_defaults |
| `--epochs` | ✅ | task 可覆盖默认 |
| `--lr` | ✅ | |
| `--batch_size` | ✅ | |
| `--save_steps` | ✅ | ablation 模式 |
| `--random_3d_seed` | ✅ | random_3d mode |
| `--instruction` | 🆕 | 覆盖 registry 中的 instruction |
| `--local_rank` | ✅ | deepspeed |

### 2.5 示例命令（目标态）

```bash
# TmQM property（原 Property.py）
deepspeed --num_gpus=2 Stage3.py \
  --task homo_lumo_gap --mode single_token \
  --train_lmdb /path/train.lmdb --val_lmdb /path/valid.lmdb \
  --Stage2_ckpt /path/Stage2

# Property ablation
deepspeed --num_gpus=2 Stage3.py \
  --task dipole_moment --mode freeze_3d \
  --3D_encoder_ckpt /path/3D_encoder.pt

# Vaska（原 Vaska_Complex.py）
deepspeed --num_gpus=2 Stage3.py \
  --task vaska_barrier --mode single_token \
  --lmdb /data/jingyuan_data/vaskas-space/data.lmdb \
  --split_seed 43 --output_dir VASKA

# NiComplex（原 NiComplex.py）
deepspeed --num_gpus=2 Stage3.py \
  --task nicomplex_ddg --mode single_token \
  --lmdb /data/jingyuan_data/NiComplex/part1.lmdb \
  --lmdb /data/jingyuan_data/NiComplex/part2.lmdb \
  --split_seed 38
```

---

## 3. 推理 CLI 扩展（`inference.py`）

统一子命令（目标态）：

```bash
python inference.py stage3 \
  --task vaska_barrier \
  --mode single_token \
  --ckpt /path/to/ckpt \
  --lmdb /path/to/test.lmdb \
  --save_json out.json
```

| 参数 | 说明 |
|------|------|
| `--task` | 同训练 |
| `--mode` | 同训练 |
| `--ckpt` | Stage3 checkpoint |
| `--model_name` / `--3D_encoder_dict` | 与现 inference 一致 |
| `--lmdb` / `--test_lmdb` | 评估数据 |
| `--instruction` | 可选覆盖 |
| `--save_json` | 输出 |
| `--max_samples` / `--batch_size` / `--gpus` | 保留 |

**PROPERTY_MODES 扩展为与训练一致的五种 mode**；description 子命令照旧，后续也从 registry 读 instruction。

---

## 4. 文件处置清单

### 4.1 合并后删除（主训练脚本）

| 文件 | 原因 |
|------|------|
| `Stage3/Property.py` | 并入 `Stage3.py` |
| `Stage3/NiComplex.py` | 并入 `Stage3.py`；Dataset 逻辑迁到 `task_datasets.py` |
| `Stage3/Vaska_Complex.py` | 同上 |

删除前把 **Dataset / LMDB reader** 迁出：

- `read_vaska_lmdb`, `VaskaComplexDataset` → `task_datasets.py`
- `read_nicomplex_lmdb`, `NiComplexDDGDataset`, `load_merged_valid_nicomplex_records` → `task_datasets.py`

### 4.2 推理脚本：删除或薄封装

| 文件 | 处置 |
|------|------|
| `inference_Vaska_Complex.py` | **删除** → `inference.py stage3 --task vaska_barrier` |
| `OOD/NiComplex/inference_nicomplex_ood.py` | **删除** → 统一 inference |
| `OOD/NiComplex/inference_nicomplex_ood_multigpu.py` | **已删除**（NiComplex OOD 测试集较小；用 `inference.py stage3`） |
| `OOD/Vaska/inference_vaska_ood.py` | **删除** |
| `OOD/Vaska/inference_vaska_ligand_ood.py` | **删除** |
| `OOD/inference_property_ood.py` | **保留或合并** 到 `inference.py`（cluster OOD 可 `--split cluster`） |

### 4.3 OOD 训练脚本：第二阶段合并

| 文件 | 处置 |
|------|------|
| `OOD/Vaska/Vaska_OOD.py` | 暂留 → 最终 `Stage3.py --task vaska_barrier --split loobo_b_group` |
| `OOD/Vaska/Vaska_Ligand_OOD.py` | 暂留 → `--split ligand_loobo` |
| `OOD/NiComplex/NiComplex_OOD.py` | 暂留 → `--split nicomplex_ood` |
| `OOD/Property_OOD.py` | 暂留 |

**保留的 OOD 工具模块（无 `if __name__` 训练入口）**

| 文件 | 用途 |
|------|------|
| `OOD/Vaska/b_group_split.py` | LOBO B-group |
| `OOD/Vaska/ligand_split.py` | Ligand OOD |
| `OOD/NiComplex/nicomplex_split.py` | NiComplex OOD split |
| `OOD/cluster_split.py` | Property cluster OOD |
| `OOD/dataset_ood.py` | 共用 OOD 数据工具 |

### 4.4 保留不动（本阶段）

| 文件 | 原因 |
|------|------|
| `Stage1.py`, `Stage2.py` | 下一阶段再接 registry |
| `task_datasets.py` | Dataset 实现；逐步删重复 instruction 常量 |
| `multimodal_LLM.py`, `utils.py` | 核心模型 |
| `train_defaults.py` | 暂留；可收敛为 `STAGE3_DEFAULTS` 单 dict + task 覆盖 |
| `run_description_ablations.sh` | Stage2 专用 |

### 4.5 Shell 脚本更新

| 脚本 | 修改 |
|------|------|
| `run_vaska_ten_splits.sh` | `Stage3/Vaska_Complex.py` → `Stage3.py --task vaska_barrier` |
| `run_vaska_ood_train_infer.sh` | 训练 → `Stage3.py`；推理 → `inference.py` |
| `run_vaska_ligand_ood_train_infer.sh` | 同上 |
| `run_nicomplex_ood_train_infer.sh` | `NiComplex.py` → `Stage3.py --task nicomplex_ddg` |
| `run_ood_property_train_infer.sh` | `Property_OOD.py` → `Stage3.py`（后续） |

### 4.6 文档 / 陈旧目录

| 路径 | 处置 |
|------|------|
| `ablation_experiments/` | 删除或 README 指向 `Stage2.py --mode` |
| `OOD/Vaska/README.md`, `OOD/README.md` | 更新命令示例 |

---

## 5. `train_defaults.py` 收敛建议

合并后 Stage3 默认可从 registry + 一个 dict 读取：

```python
STAGE3_DEFAULTS = {
    **PROPERTY_DEFAULTS,
    "task": "homo_lumo_gap",
    "mode": "single_token",
}
# 删除独立 NICOMPLEX_DEFAULTS / VASKA_DEFAULTS 中的重复字段；
# task 专属 default_lmdb / split_seed 移到 task_registry.TaskSpec 或 TASK_TRAIN_DEFAULTS[task]
```

---

## 6. 实施顺序（建议）

1. **落地 `task_registry.py`**（已完成）+ 单测 `resolve_user_content` 与现有 Dataset 输出对齐  
2. **`task_datasets.py`**（已完成）：PROPERTY_CONFIG / INSTRUCTION_* 从 registry 导入；Vaska/NiComplex Dataset 已迁入；`Stage3/NiComplex.py` / `Vaska_Complex.py` 改为 re-export + 训练入口  
3. **新建 `Stage3.py`**（已完成）：统一 `--task` + `--mode`；`Stage3/Property.py` 等为薄封装  
4. **扩展 `inference.py`**（已完成）：`stage3` 子命令；Property CSV split；Vaska 26-ligand OOD；NiComplex Pybox/Biox/Biim  
5. **更新 shell + 删旧脚本**（已完成 b_group 删除；旧薄封装仍可用）  
6. **OOD 训练并入** `--split`（可选第二波）  
7. **Stage1/2 接 registry**（第三波）

---

## 7. 风险与注意点

- **NiComplex instruction** 含 `{smiles}`、`{temp}` 占位符：必须用 `resolve_user_content`，不能简单 `instruction + smiles`。  
- **Vaska** 要求 LMDB 有非空 `smiles`；**NiComplex** 的 temp 缺失时需定义默认（当前 format 为空字符串）。  
- **Property ablation** 与 **Vaska/NiComplex 主路径** 训练超参不同（grad_accum、eval_strategy、deepspeed）：`Stage3.py` 内按 `task` + `mode` 保留 `_training_args` 分支表。  
- **wandb project 名**：用 `task_registry.wandb_project()` 保持与历史实验可比。  
- **`Stage3/Property.py` 的 `__all__` 导出**：`OOD/Property_OOD.py` 等 `from Property import PROPERTY_CONFIG` → 改为 `from task_registry import property_config_dict` 或 `from task_datasets import PROPERTY_CONFIG`。

---

## 8. 验收标准

- [ ] 五条训练路径 smoke test 通过：`dipole_moment`, `homo_lumo_gap`, `vaska_barrier`, `nicomplex_ddg`, `dipole_moment --mode freeze_3d`  
- [ ] 旧 shell 脚本仅改路径即可跑通  
- [ ] `inference.py` 可评估 TmQM property + Vaska + NiComplex  
- [ ] 全仓库无重复 instruction 字面量（除 `task_registry.py`）  
- [ ] `python -m py_compile Stage3.py task_registry.py` 通过
