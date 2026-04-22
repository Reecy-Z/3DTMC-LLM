# 3DTMC-LLM

**3DTMC-LLM: Bridging 3D Geometry and Large Language Models for Transition Metal Complexes**

![Graphical abstract](abstract_graphic.png)

This repository implements a **3D encoder** fused with a **causal LLM** — by default **[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** — for generative tasks on transition metal complexes (TMCs). The stack is trained in stages: **3D encoder pretraining → Stage 1 (frozen LLM) → Stage 2 (LLM + LoRA + continued 3D alignment)**, then **downstream fine-tuning** on property prediction, barrier regression, and related tasks.

### Environment

```bash
pip install -r requirements.txt
```

#### Uni-Core

Training and inference import **`unicore`** (e.g. `Dictionary` and the distributed framework). Install **Uni-Core** from source or wheels as documented in the upstream repo:

**[https://github.com/dptech-corp/Uni-Core](https://github.com/dptech-corp/Uni-Core)**

---

## Models and Datasets

Checkpoints and Stage 1 data for **3DTMC-LLM** are published on Hugging Face under **[Reecy/3DTMC-LLM](https://huggingface.co/Reecy/3DTMC-LLM)**.

| Asset | Location |
|------|----------|
| **Stage 2** (trained stack: LoRA adapter, tokenizer files, `3D_encoder.pt`, trainer state, etc.) | [Stage2](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/Stage2) |
| **3D encoder (pretrained)** | [3D_encoder_pretrain](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/3D_encoder_pretrain) |
| **Atom vocabulary for the 3D encoder** (`dict.txt` format) | [3D_encoder_dict.txt](https://huggingface.co/Reecy/3DTMC-LLM/blob/main/3D_encoder_dict.txt) |
| **Stage 1 — TMC-Prop3D** (LMDB) | [TMC-Prop3D.lmdb](https://huggingface.co/Reecy/3DTMC-LLM/blob/main/TMC-Prop3D.lmdb) |

**Building datasets / text corpora**

- **`enrich_description.py`** — starting from an LMDB with **`description`** and **`smiles`**, calls an OpenAI-compatible LLM and writes polished text to **`enriched_description`**. Use this workflow to build a **TMC-Prop3D-Enriched** dataset (enriched descriptions in LMDB) for Stage 2 training.
- **`generate_QA_pairs.py`** — loads knowledge-source files (e.g. PDF, TXT, Markdown), splits into chunks, and uses the Chat Completions API to generate **Q&A pairs** saved for Stage 2 training.

---

## 1. 3D encoder

Downstream **Stage 1 / 2 / Property / NiComplex / Vaska** code expects **pretrained 3D encoder weights** and a **`dict.txt`** atom vocabulary.

### Option A — use our pretrained encoder

Load the encoder from **[Reecy/3DTMC-LLM/3D_encoder_pretrain](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/3D_encoder_pretrain)** and use atom vocabulary **[3D_encoder_dict.txt](https://huggingface.co/Reecy/3DTMC-LLM/blob/main/3D_encoder_dict.txt)** (as `dict.txt`).

### Option B — pretrain the 3D encoder yourself

1. **Data**  
   Use the **OMol25 from Meta**, e.g. **[`facebook/OMol25`](https://huggingface.co/datasets/facebook/OMol25)** or other TMC datasets, to build a large corpus of **TMC 3D structures**.  

2. **Vocabulary**  
   Obtain or build **`dict.txt`**.

3. **Training**  
   Run **`3D_encoder_trainer.py`**, using the **masked-atom, coordinate, and pairwise-distance** pretraining objective:

   ```bash
   # Example: adjust paths to your LMDB train/valid, dict.txt, and DeepSpeed JSON.
   deepspeed --num_gpus=N 3D_encoder_trainer.py \
     --dict /path/to/dict.txt \
     --train-path /path/to/train_lmdb_or_dir \
     --valid-path /path/to/valid.lmdb \
     --output-dir /path/to/encoder_pretrain_out \
     --deepspeed /path/to/deepspeed_config.json \
     --max-steps ... --save-steps ... --eval-steps ...
   ```

---

## 2. Stage 1

**Script:** `Stage1.py`  

- Trains the **3D encoder + single-token projection** into the LLM embedding space; **LLM is frozen**.  
- **Data:** tmQM-style **LMDB** with `atoms`, `coordinates`, `smiles`, `description`.  
- **Run (example):**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Stage1.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --train_lmdb /path/train.lmdb /path/valid.lmdb --val_lmdb /path/valid.lmdb --output_dir ...
  ```

  Use the same **`dict.txt`** and **encoder checkpoint** paths as in `train_defaults.py` / your HF download.

Defaults are centralized in **`train_defaults.py`** (`STAGE1_DEFAULTS`).

---

## 3. Stage 2

**Script:** `Stage2.py`  

- Continues from a **Stage 1 checkpoint**: **LoRA** on the LLM, **3D encoder + single-token projection** trainable.  
- **Mixed data:** (1) **LMDB** with `enriched_description`; (2) **JSON Q&A**.  

**Run (example):**

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Stage2.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 --Stage1_ckpt /path/to/stage1_checkpoint \
  --train_lmdb ... --val_lmdb ... --json_qa ... --output_dir ...
```

Pretrained **Stage 2** weights are provided under **[Reecy/3DTMC-LLM/Stage2](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/Stage2)** (see **Models and Datasets** above).

Defaults: **`train_defaults.py`** (`STAGE2_DEFAULTS`).

---

## 4. Stage 3 (Downstream tasks)

**`Stage3/Property.py`**, **`Stage3/NiComplex.py`**, and **`Stage3/Vaska_Complex.py`** initialized from a **Stage 2 `Stage2_ckpt`** (HF adapter + encoder weights + single-token projection weights).

For **Vaska_Complex**, we provide a **ready-made LMDB** (`data.lmdb`) on the Hub at **[Reecy/3DTMC-LLM/vaskas-space](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/vaskas-space)** so you can run the task.

For **NiComplex** data preparation, we provide **`datasets_generation/build_ni_complex.py`**: it uses **MetalloGen** to **assemble a five-coordinate square-pyramidal Ni complex** from a **bidentate nitrogen-donor ligand** (`Ligand*_N*.xyz`, with donor indices in the filename) and three **substrate** fragments (`R1_stay*.xyz`, `R2_stay*.xyz`, `R2_leave*.xyz`) in a single case directory, and writes **`complex_Ni.xyz`** there. Requires Gaussian/xtb setup as in the script header (e.g. `xtbbin`). Example:

```bash
python datasets_generation/build_ni_complex.py datasets_generation/NiComplex_example
```

| Task | Script | Dataset / reference |
|------|--------|---------------------|
| Quantum properties (dipole, polarisability, HOMO–LUMO, …) | `Stage3/Property.py` | **[tmQMg](https://github.com/hkneiding/tmqmg)** — graph/property release and analysis (see also [uiocompcat/tmQMg](https://github.com/uiocompcat/tmQMg) for the dataset family). |
| H₂ activation energy barrier (Vaska-type Ir complexes) | `Stage3/Vaska_Complex.py` | Original release: **[vaskas-space](https://github.com/pascalfriederich/vaskas-space)**. Prebuilt LMDB for this workflow: **[`vaskas-space/data.lmdb`](https://huggingface.co/Reecy/3DTMC-LLM/tree/main/vaskas-space)** on Hugging Face. |
| Ni-catalyzed enantioselective coupling (ΔΔG / related targets) | `Stage3/NiComplex.py` | **[Enantioselective-Cross-Coupling-Prediction](https://github.com/TheLiaoGroup/Enantioselective-Cross-Coupling-Prediction)** |

To **test Vaska barrier** with **10 different random 80/10/10 splits**, run from the repo root:

```bash
bash run_vaska_ten_splits.sh
```

The script trains **Stage3/Vaska_Complex.py** (2 GPUs) for each seed, runs **`inference_Vaska_Complex.py`**, and writes one JSON per seed under **`Vaska_Complex_Results/`** (e.g. `pred_vaska_barrier_seed_38.json`). Seeds whose prediction file already exists and is non-empty are **skipped** so you can resume. Paths and hyperparameters come from **`train_defaults.py`** (`VASKA_DEFAULTS`, including LMDB and `Stage2_ckpt`); adjust there or pass overrides if your fork uses different locations. At the end it runs **`plot_vaska_barrier.py`** on the collected JSON files.

For a **single** split without the loop, call **`Stage3/Vaska_Complex.py`** directly, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 Stage3/Vaska_Complex.py --split_seed 38
```

Hyperparameters and paths: **`train_defaults.py`** (`PROPERTY_DEFAULTS`, `NICOMPLEX_DEFAULTS`, `VASKA_DEFAULTS`).

---

## 5. Interactive inference demo

**`inference_demo.py`** — single-structure **SMILES + XYZ + free-form instruction**. Example:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --Stage2_ckpt /path/to/checkpoint \
  --smiles "..." \
  --xyz /path/to/structure.xyz \
  --instruction "Your question here."
```

---

## 6. Building TMC structures and SMILES

| Goal | Resource |
|------|----------|
| **XYZ → SMILES** for TMCs | **[jensengroup/xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm)** |
| **Generate / assemble 3D TMC conformers** | **[kyunghoonlee777/MetalloGen](https://github.com/kyunghoonlee777/MetalloGen)** |

---

## 7. Repository layout (core)

| File | Role |
|------|------|
| `3D_encoder_trainer.py` | 3D encoder LMDB pretraining with `Trainer` + DeepSpeed |
| `multimodal_LLM.py` | Geometry + Qwen stack; defines single-token-projected 3D tokens + LLM (Stage 1 / 2 / 3 recipes) |
| `utils.py` | LMDB helpers, 3D batching, single-token embed fusion, collators |
| `Stage1.py` / `Stage2.py` | Instruction-tuning stages |
| `Stage3/Property.py` / `Stage3/NiComplex.py` / `Stage3/Vaska_Complex.py` | Downstream SFT |
| `inference_demo.py` | Generic single-token + 3D generation demo |
| `train_defaults.py` | Default paths and hyperparameters |
| `datasets_generation/enrich_description.py` | LMDB → LLM polish → `enriched_description` (TMC-Prop3D-Enriched build) |
| `datasets_generation/generate_QA_pairs.py` | Knowledge sources → chunked Q&A JSON |
| `datasets_generation/build_ni_complex.py` | MetalloGen: assemble a five-coordinate Ni complex XYZ|
| `requirements.txt` | Python dependencies (after PyTorch) |

---

## Citation

If you use **3DTMC-LLM**, please cite the **3DTMC-LLM** paper when available. 

---

## License

- **This repository** (code layout, training scripts, and assets **you** publish under **[Reecy/3DTMC-LLM](https://huggingface.co/Reecy/3DTMC-LLM)**): follow the **LICENSE** file in this repo and the terms on each Hugging Face model/data card you upload.
- **Third-party (not authored here):** **[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** and **[OMol / OMol25](https://huggingface.co/datasets/facebook/OMol25)** (and related OMol resources) are separate products with **their own** licenses, attribution, and use restrictions—obey those sources, not this README alone.
- **Other** linked datasets, tools, and **baseline 3D encoder** code: each has its own license; check upstream repos and papers.