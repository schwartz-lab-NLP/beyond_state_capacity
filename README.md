# Beyond State Capacity: Why Retrieval Breaks in Mamba

This repository contains the experimental code accompanying the paper **Beyond State Capacity: Why Retrieval Breaks in Mamba**. It implements the evaluations, probes, and causal interventions reported in the paper, and writes raw results and diagnostic plots that can be used to regenerate the published figures.

## Hardware Requirements

All experiments were conducted on a single NVIDIA L40S GPU (48 GB VRAM), except for the state probing experiment, which used two GPUs: one for model inference and state extraction, and the other for training the probes. Across both models, both datasets, and all experiments presented in this work, the total compute budget was roughly 1,000 single-GPU hours.

A CUDA 12.1+ toolkit and a compatible NVIDIA driver are required.

## Environment Variables

None of the path or cache variables below are required for a first run: unset variables fall back to the defaults listed here. `CUDA_HOME` and `LD_LIBRARY_PATH` must be set correctly before installing the Mamba kernels (see [Environment Setup](#environment-setup)).

### Output paths

All Python code resolves output locations in [`libs/paths.py`](libs/paths.py). Defaults are repository-relative (`<repo>/output_dir`). More specific variables take precedence over `OUTPUT_DIR`.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OUTPUT_DIR` | No | `<repo>/output_dir` | Umbrella root used when the three variables below are unset |
| `RAW_RESULTS_PATH` | No | `$OUTPUT_DIR` | Experiment JSON / parquet artifacts |
| `PLOTS_PATH` | No | `$OUTPUT_DIR` | Diagnostic PNG plots |
| `FIGURES_PATH` | No | `$OUTPUT_DIR/figures` | Paper PDFs written by `figures_generation/` |

On-disk layout:

```
$RAW_RESULTS_PATH/<model>_<task>/<experiment>/<dataset>/raw_results/
$PLOTS_PATH/<model>_<task>/<experiment>/<dataset>/plots/
$FIGURES_PATH/<name>.pdf
```

### Model weights and caches

Checkpoints are downloaded into Hugging Face's default cache unless overridden.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `HF_HOME` | No | Hugging Face default | Hugging Face home (cache, tokens, etc.) |
| `HF_CACHE_DIR` | No | Hugging Face default | Cache directory for Hugging Face Mamba-2 / Codestral loads |
| `HF_HUB_OFFLINE` | No | unset (online) | Set to `1` to use a prefetched cache without network access |
| `NVIDIA_MAMBA_WEIGHTS_DIR` | No | Hugging Face cache | If set, NVIDIA Mamba weights and tokenizer are stored here |

Gated Hugging Face repositories may require an authenticated session (`huggingface-cli login`) before the first download.

### Runtime variables set by the entry point

[`main.py`](main.py) sets `PYTORCH_CUDA_ALLOC_CONF` to `expandable_segments:True` if that variable is not already defined. When `--debug` is passed, it also sets `CUDA_LAUNCH_BLOCKING=1`. Neither variable needs to be exported by the user.

## Environment Setup

### Prerequisites

- CUDA 12.1+ (CUDA Toolkit must be installed on the system)
- Python 3.11

### 1. CUDA environment variables

Ensure `CUDA_HOME` and `LD_LIBRARY_PATH` are defined. If they are not:

```bash
export CUDA_HOME=/usr/local/nvidia/cuda/12.8.1  # Adjust to your CUDA path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Create the virtual environment

```bash
# 1. Create and activate
python3 -m venv .venv
source .venv/bin/activate

# 2. Essential build tools (setuptools must be < 70 for the Mamba build)
pip install "setuptools<70" wheel packaging ninja

# 3. PyTorch and core dependencies (matches CUDA 12.1)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" einops triton==2.1.0

# 4. Mamba kernels (installed from source to ensure C++ linking)
pip install "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" --no-build-isolation --no-deps
pip install "git+https://github.com/state-spaces/mamba.git@v2.2.2" --no-build-isolation --no-deps

# 5. Specific Transformers version (newer versions clash with mamba_ssm)
pip install transformers==4.45.0

# 6. Remaining requirements
pip install accelerate faker matplotlib pandas sentencepiece scipy scikit-learn seaborn pyarrow fastparquet

# 7. Verify
python -c "import torch; import transformers; from mamba_ssm import Mamba; print('OK')"
```

## Experiments

Each experiment is launched through [`main.py`](main.py) and implemented under [`libs/experiments/`](libs/experiments/). Command-line defaults are defined in [`libs/args.py`](libs/args.py). Unless noted otherwise, the commands below use those defaults (`--model mamba_nvidia`, `--task prefix`, `--dataset phonebook`, `--seed 42`).

Paper results use both models (`--model mamba_nvidia` and `--model mamba_codestral`). Repeat or edit the commands as needed.

### Evaluation (Section 4.1)

Measures generation accuracy as a function of directory length on the selected retrieval task.

```bash
python main.py --experiment evaluation --dataset phonebook
python main.py --experiment evaluation --dataset phonebook_local
```

### State probing (Section 4.2)

Trains per-layer linear probes on the recurrent SSM state to test whether it stores the target entry. This experiment requires `profbook` or `profbook_local`. The paper used two GPUs (inference on one device, probe training on the other).

```bash
python main.py --experiment state_probing --dataset profbook --state-probing-num-gpus 2
python main.py --experiment state_probing --dataset profbook_local --state-probing-num-gpus 2
```

### Selection distinction (Section 4.3)

Records an input-dependent mixer component at every token to test whether the target entry is treated differently from distractors.

```bash
python main.py --experiment selection_distinction --dataset phonebook
python main.py --experiment selection_distinction --dataset phonebook_local
```

### Patching and scaling (Sections 5, 6.3)

Performs causal activation-patching and in-place scaling of mixer components to test whether they account for the accuracy difference between local and global prompts.

```bash
python main.py --experiment patching_and_scaling --patch-donor local --patch-recipient global
python main.py --experiment patching_and_scaling --patch-donor global --patch-recipient local --sweep all
python main.py --experiment patching_and_scaling --patch-donor scaling --patch-recipient global --patch-scale 0.8 --sweep all
python main.py --experiment patching_and_scaling --patch-donor scaling --patch-recipient global --patch-scale 1.2 --sweep all
python main.py --experiment patching_and_scaling --patch-donor scaling --patch-recipient global --patch-scale 1.5 --sweep all
python main.py --experiment patching_and_scaling --patch-donor scaling --patch-recipient global --patch-scale 2.0 --sweep all
python main.py --experiment patching_and_scaling --patch-donor scaling --patch-recipient global --patch-scale 3.0 --sweep all
python main.py --experiment patching_and_scaling --patch-donor local --patch-recipient global ----patching-and-scaling-components u
```

### Representation probing (Section 6.1)

Trains per-layer linear probes on mixer input representations to test whether the target entry is linearly decodable.

```bash
python main.py --experiment representation_probing --dataset phonebook
python main.py --experiment representation_probing --dataset phonebook_local
```

### Gate probing (Section 6.2)

Trains per-layer linear probes on selectivity gates (`dt_raw` / `delta`) to test whether they encode target identity.

```bash
python main.py --experiment gate_probing --dataset phonebook
python main.py --experiment gate_probing --dataset phonebook_local
```

## Figures

Paper figures are generated from the saved experiment artifacts under `$RAW_RESULTS_PATH` by the scripts in [`figures_generation/`](figures_generation/). PDFs are written to `$FIGURES_PATH`. The scripts import both `libs` and sibling modules in `figures_generation/`, so `PYTHONPATH` must include the repository root and that directory.

```bash
# Figure 2
PYTHONPATH=.:figures_generation python figures_generation/correlation.py --model mamba_nvidia

# Figure 3
PYTHONPATH=.:figures_generation python figures_generation/combined_causality.py

# Figure 4
PYTHONPATH=.:figures_generation python figures_generation/combined_failure.py

# Figure 5
PYTHONPATH=.:figures_generation python figures_generation/correlation.py --model mamba_codestral
```
