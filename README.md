# PE-Sleuth: Program-level Semantics + Static Feature Fusion for Interpretable Ransomware Detection with LLMs

PE-Sleuth decompiles Windows PE binaries to C, derives program-level semantics, fuses them with compact static features, and uses an LLM to perform binary classification (RANSOMWARE vs. BENIGN) with an analyst-oriented rationale. The code supports both a zero-shot base model and a LoRA-fine-tuned variant, consistent with the paper’s terminology and pipeline.

> Paper title: *PE-Sleuth: Program-level Semantics and Static Feature Fusion for Interpretable Ransomware Detection with LLMs*
> Keywords: Ransomware Detection, Portable Executable Files, Program-level Semantics, Static Features, Large Language Models

---

## Project structure

```
PE-Sleuth/
├─ Classify/
│  ├─ Classify_Input_C_Code/                      # input .c files (empty by default)
│  └─ Classify_Output_All/                        # outputs (empty by default)
│     ├─ 1a_parsed_metadata/                      # normalized static features
│     ├─ 3_visualizations/                        # call-graph/block visualizations
│     ├─ 5_program_summaries/                     # program-level semantics
│     ├─ 7_classification_results/                # predictions
│     └─ 8_classification_rationale/              # concise rationale
├─ Decompile/
│  ├─ Decompile_Input_Raw_PE/                     # input PE binaries (empty by default)
│  └─ Decompile_Output_C_Code/                    # exported .c files
│
├─ LLMs/
│  ├─ Base_Model/
│  │  └─ Qwen3-14B/                               # clone base model here
│  └─ LoRA_Weight/
│     └─ PE-Sleuth-Qwen3-14B-LoRA/                # clone LoRA adapter here
│
├─ batch_decompile_ida.py                         # decompilation driver (IDA/Hex-Rays)
├─ classify_from_c_code.py                        # end-to-end pipeline: fusion + LLM
├─ model_server.py                                # local LLM loader / service (optional)
└─ readme.md
```

> Note: Additional helper files/folders may be created during runs (e.g., intermediate IR, call graphs).

---

## Platform & requirements

* Decompilation: Windows only (requires IDA Pro 9+ with Hex-Rays).
* Subsequent stages (feature extraction, semantics, classification, rationale, visualization): Windows or Linux.

Python: 3.10+
Recommended packages: `torch`, `transformers`, `peft`, `bitsandbytes` (4-bit `nf4`), and optionally FlashAttention if supported by your GPU.

Models (local only):

* Base model: Qwen3-14B
* Task-adaptive LoRA: PE-Sleuth-Qwen3-14B-LoRA

---

## Installation

1. Clone this repository and create a fresh virtual environment.

2. Install dependencies (example):

```bash
pip install torch transformers peft bitsandbytes
# (optional) flash-attn per your CUDA/SM capability
```

3. Prepare local models (no external API calls):

```bash
# Base model
cd LLMs/Base_Model
git clone https://huggingface.co/Qwen/Qwen3-14B

# LoRA adapter
cd ../../LLMs/LoRA_Weight
git clone https://huggingface.co/AlexAshlake/PE-Sleuth-Qwen3-14B-LoRA
```

4. Install IDA Pro (Windows) and set `IDA_PATH` to the folder containing `idat.exe` or to the full path of `idat.exe`.
   *(Required only for the decompilation step.)*

```bat
:: Windows examples
setx IDA_PATH "C:\Program Files\IDA Pro 9.1"
:: or
setx IDA_PATH "C:\Program Files\IDA Pro 9.1\idat.exe"
```

5. Dataset (for reproducing the paper):
   Ransomware PE Header Feature Dataset (Mendeley v3): [https://data.mendeley.com/datasets/p3v94dft2y/3]

---

## Quick start

### 1) Decompile PE → C  (Windows only)

1. Place Windows PE binaries in:

   ```
   Decompile/Decompile_Input_Raw_PE/
   ```
2. Run:

   ```bash
   python batch_decompile_ida.py
   ```
3. The script invokes Hex-Rays via `idat.exe` and writes `.c` files to:

   ```
   Decompile/Decompile_Output_C_Code/
   ```
4. Abnormal-size safeguard: if an exported `.c` is < 10 KB or > 3 MB, the file is deleted and the sample is recorded as a failed decompilation (see the decompilation log).

   * A log file (e.g., `decompile_failure_log.txt`) in `Decompile/` summarizes failures.

### 2) Classify & generate rationale  (Windows or Linux)

1. Ensure your `.c` files are in:

   ```
   Classify/Classify_Input_C_Code/
   ```
2. Run:

   ```bash
   python classify_from_c_code.py
   ```
3. The script performs the full pipeline consistent with the paper:

   * Static Features Extraction → `1a_parsed_metadata/`
   * Semantic Abstraction (call-graph-aware partitioning, block → program)
   * Fusion & Classification → `7_classification_results/`
   * Rationale Generation → `8_classification_rationale/`
   * Visualizations (FCG & blocks) → `3_visualizations/`

> By default the pipeline loads the base model from `LLMs/Base_Model/Qwen3-14B/` and applies the LoRA adapter from `LLMs/LoRA_Weight/PE-Sleuth-Qwen3-14B-LoRA/` if present, reflecting the “Fine-tuned-Full” configuration in the paper. To run a zero-shot baseline, remove/disable the LoRA adapter.

---

## Reproducing the paper’s experiments

1. Download the Mendeley dataset (v3 binaries) and place the samples under `Decompile/Decompile_Input_Raw_PE/`.
2. Decompile on Windows to produce `.c` sources.
3. Run the classification pipeline on Windows or Linux to generate static features, program-level semantics, fused IR, predictions, and rationales under `Classify/Classify_Output_All/`.
4. Evaluation settings:

   * Base-Full (zero-shot): use only the base model.
   * Fine-tuned-Full: load the LoRA adapter (recommended for highest recall + stability).
   * For ablations, feed only program-level semantics or only static features.

---

## Troubleshooting

* `idat.exe` not found / `IDA_PATH` issues (Windows): set `IDA_PATH` to the folder containing `idat.exe` or to the full path.
* Running decompilation on Linux: not supported. Perform Step 1 on Windows, then copy the generated `.c` files to your Linux machine for Step 2.
* GPU memory pressure: default 4-bit quantization (`bitsandbytes`, `nf4`) reduces VRAM use. Shorten contexts or batch sizes if needed.

---

## Methodology (terminology aligned with the paper)

* Decompile: batch export Hex-Rays C from PE binaries.
* Semantic Abstraction: call-graph–aware partitioning, block-level behavior extraction, distillation to program-level semantics.
* Static Features Extraction: normalized global cues (imports, strings, declarations, indicators) under a strict budget.
* Classification & Rationale: fuse semantics + static features; output a single label and a concise analyst rationale linking evidence to behavior.

---

## Citation

If you use this code or the LoRA weights, please cite:

PE-Sleuth: Program-level Semantics and Static Feature Fusion for Interpretable Ransomware Detection with LLMs

---

## License

This repository is released for academic reproducibility. See LICENSE for terms.
