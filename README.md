# 3DTMC-LLM

**3DTMC-LLM: Bridging 3D Geometry and Large Language Models for Transition Metal Complexes**

![Graphical abstract](abstract_graphic.png)

This repository implements a **3D encoder** (BOS / global geometry token) fused with a **causal LLM** — by default **[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** — for generative tasks on transition metal complexes (TMCs). The stack is trained in stages: **3D encoder pretraining → Stage 1 (frozen LLM) → Stage 2 (LLM + LoRA + continued 3D alignment)**, then **downstream fine-tuning** on property prediction, barrier regression, and related tasks.

### Environment

```bash
pip install -r requirements.txt
```

#### Uni-Core

Training and inference import **`unicore`** (e.g. `Dictionary` and the distributed framework). Install **Uni-Core** from source or wheels as documented in the upstream repo:

**[https://github.com/dptech-corp/Uni-Core](https://github.com/dptech-corp/Uni-Core)**

---

## Models and Datasets

- **This project’s 3D + Stage 2 assets:** pretrained **3D encoder**, **Stage 2** checkpoints, and associated **training data** on **[Reecy/TMC](https://huggingface.co/Reecy/TMC)**.

Download the weights and data from those collections and point the training scripts at local paths (or adapt loaders to read from the Hub).

---

## 1. 3D encoder

Downstream **Stage 1 / 2 / Property / NiComplex / Vaska** code expects **pretrained 3D encoder weights** and a **`dict.txt`** atom vocabulary consistent with your OMol / tmQM pipeline.

### Option A — use our pretrained encoder

Load the encoder weights from **[Reecy/TMC](https://huggingface.co/Reecy/TMC)** and pass the path via the **Stage 1 geometry-encoder argument** (see `Stage1.py --help`). For Stage 3–style scripts, ensure the **encoder state file** inside `init_ckpt` matches that release. Use the **`dict.txt`** bundled with the same OMol25 / training setup (see `train_defaults.py` for typical paths).

### Option B — pretrain the 3D encoder yourself

1. **Data**  
   Use the **OMol25-scale molecular data from Meta on Hugging Face**, e.g. **[`facebook/OMol25`](https://huggingface.co/datasets/facebook/OMol25)** (and related OMol resources), to build a large corpus of **3D structures**.  
   Convert the release into **LMDB shards** where each record is at least a **`dict` with `atoms` and `coordinates`** (and optional metadata). Atom entries should be mappable to your **`dict.txt`** (element symbols or atomic numbers; see `3D_Encoder_Trainer.py` for integer→symbol handling via RDKit).

2. **Vocabulary**  
   Obtain or build **`dict.txt`** in the same **line-per-token** format as standard **OMol25_MC** atom vocabularies (one token per line; special symbols such as `[MASK]` are added in code).

3. **Training**  
   Run **`3D_Encoder_Trainer.py`** with HuggingFace `Trainer` + **DeepSpeed**, using the **masked-atom, coordinate, and pairwise-distance** pretraining objective:

   ```bash
   # Example: adjust paths to your LMDB train/valid, dict.txt, and DeepSpeed JSON.
   deepspeed --num_gpus=N 3D_Encoder_Trainer.py \
     --dict /path/to/dict.txt \
     --train-path /path/to/train_lmdb_or_dir \
     --valid-path /path/to/valid.lmdb \
     --output-dir /path/to/encoder_pretrain_out \
     --deepspeed /path/to/deepspeed_config.json \
     --max-steps ... --save-steps ... --eval-steps ...
   ```

   A reference DeepSpeed fragment for bf16/ZeRO is in **`ds_config_3D_Encoder.json`** in this folder; merge or align it with your full **`deepspeed_config.json`** (batch sizes, optimizer, etc.). The script logs to **Weights & Biases** by default (`--no-wandb` to disable).

4. **Output**  
   Saved checkpoints are standard **HuggingFace `Trainer` checkpoints** containing the **encoder** weights. Point **Stage 1** at that checkpoint directory (or exported weights) as in `train_defaults.py`. For later stages, place the **encoder state** and **BOS projection** tensors in the layout expected by your `init_ckpt` (see any released checkpoint on **[Reecy/TMC](https://huggingface.co/Reecy/TMC)** as a template).

---

## 2. Stage 1 — BOS projection, frozen LLM

**Script:** `Stage1.py`  

- Trains the **3D encoder + single-token BOS projection** into the LLM embedding space; **LLM is frozen**.  
- **Data:** tmQM-style **LMDB** with `atoms`, `coordinates`, `smiles`, `enriched_description`.  
- **Run (example):**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Stage1.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --train_lmdb /path/train.lmdb /path/valid.lmdb --val_lmdb /path/valid.lmdb --output_dir ...
  ```

  Use the same **`dict.txt`** and **encoder checkpoint** paths as in `train_defaults.py` / your HF download.

- Requires **`ds_config.json`** in this directory (or symlink) for DeepSpeed, consistent with your cluster. A template may live at the parent repo root (`../ds_config.json`).

Defaults are centralized in **`train_defaults.py`** (`STAGE1_DEFAULTS`).

---

## 3. Stage 2 — LLM + LoRA + mixed 3D / text

**Script:** `Stage2.py`  

- Continues from a **Stage 1 checkpoint**: **LoRA** on the LLM, **3D encoder + BOS projection** trainable.  
- **Mixed data:** (1) **LMDB** with `polished_description` and 3D fields; (2) **JSON Q&A** (text-only batches).  

**Run (example):**

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Stage2.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 --adapter /path/to/stage1_checkpoint \
  --train_lmdb ... --val_lmdb ... --json_qa ... --output_dir ...
```

Pretrained **Stage 2** weights and training corpora are provided on **[Reecy/TMC](https://huggingface.co/Reecy/TMC)**.

Defaults: **`train_defaults.py`** (`STAGE2_DEFAULTS`).

---

## 4. Downstream tasks

**`Property.py`**, **`NiComplex.py`**, and **`Vaska_Complex.py`** use the **BOS-only generative** recipe: **LoRA + 3D encoder + BOS projection**, initialized from a **Stage 2–style `init_ckpt`** (HF adapter + encoder weights + BOS projection tensors). The implementation lives in the shared **geometry + Qwen** model module imported by all training scripts.

| Task | Script | Dataset / reference |
|------|--------|---------------------|
| Quantum properties (dipole, polarisability, HOMO–LUMO, …) | `Property.py` | **[tmQMg](https://github.com/hkneiding/tmqmg)** — graph/property release and analysis (see also [uiocompcat/tmQMg](https://github.com/uiocompcat/tmQMg) for the dataset family). |
| H₂-splitting barrier (Vaska-type Ir complexes) | `Vaska_Complex.py` | **[vaskas-space](https://github.com/pascalfriederich/vaskas-space)** |
| Ni-catalyzed enantioselective coupling (ΔΔG / related targets) | `NiComplex.py` | **[Enantioselective-Cross-Coupling-Prediction](https://github.com/TheLiaoGroup/Enantioselective-Cross-Coupling-Prediction)** |

Typical invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Property.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 --property homo_lumo_gap --init_ckpt ...
```

Hyperparameters and paths: **`train_defaults.py`** (`PROPERTY_DEFAULTS`, `NICOMPLEX_DEFAULTS`, `VASKA_DEFAULTS`).

---

## 5. Interactive inference demo

**`inference_demo.py`** — single-structure **SMILES + XYZ + free-form instruction**, with 3D injected at the `object_ref` / BOS slot (same template as training). Example:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --init_ckpt /path/to/checkpoint \
  --smiles "..." \
  --xyz /path/to/structure.xyz \
  --instruction "Your question here."
```

---

## 6. Building TMC structures and SMILES

| Goal | Resource |
|------|----------|
| **XYZ → SMILES** for TMCs (Hückel / NBO / CSD workflows) | **[jensengroup/xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm)** |
| **Generate / assemble 3D TMC conformers** (m-SMILES, QC backends) | **[kyunghoonlee777/MetalloGen](https://github.com/kyunghoonlee777/MetalloGen)** |

---

## 7. Repository layout (core)

| File | Role |
|------|------|
| `3D_Encoder_Trainer.py` | 3D encoder LMDB pretraining with `Trainer` + DeepSpeed |
| Geometry + Qwen stack (imported everywhere) | Defines BOS-projected 3D tokens + LLM (Stage 1 / 2 / 3 recipes) |
| `utils.py` | LMDB helpers, 3D batching, BOS embed fusion, collators |
| `Stage1.py` / `Stage2.py` | Instruction-tuning stages |
| `Property.py` / `NiComplex.py` / `Vaska_Complex.py` | Downstream SFT |
| `inference_demo.py` | Generic BOS + 3D generation demo |
| `train_defaults.py` | Default paths and hyperparameters |
| `requirements.txt` | Python dependencies (after PyTorch) |

---

## Citation

If you use **3DTMC-LLM**, please cite the **3DTMC-LLM** paper when available. 

---

## License

- **This repository** (code layout, training scripts, and assets **you** publish under **[Reecy/TMC](https://huggingface.co/Reecy/TMC)**): follow the **LICENSE** file in this repo and the terms on each Hugging Face model/data card you upload.
- **Third-party (not authored here):** **[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** and **[OMol / OMol25](https://huggingface.co/datasets/facebook/OMol25)** (and related OMol resources) are separate products with **their own** licenses, attribution, and use restrictions—obey those sources, not this README alone.
- **Other** linked datasets, tools, and **baseline 3D encoder** code: each has its own license; check upstream repos and papers.