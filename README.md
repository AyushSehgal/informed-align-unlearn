# Informed Align-then-Unlearn

IDL Project, Spring 2026. Forked from [ExplainableML/align-then-unlearn](https://github.com/ExplainableML/align-then-unlearn) (Spohn et al., _ICML 2025 MUGen_, [arXiv:2506.13181](https://arxiv.org/abs/2506.13181)).

We extend Align-then-Unlearn with **layer-informed** unlearning: causal tracing picks the hook layer, masked losses restrict the unlearn signal to subject tokens, and a chained mode lets us forget multiple targets back-to-back.

---

## Quickstart

```bash
# 1. Clone + install (editable, pulls torch/lightning/transformers)
git clone <this-repo> informed-align-unlearn
cd informed-align-unlearn
pip install -e .

# 2. Download the RWKU benchmark data
bash data/rwku/download_rwku_data.sh

# 3. (Optional) Set your wandb entity in config/train.yaml, or just
#    export WANDB_MODE=offline to log locally.

# 4. Run it — one script, one flag.
bash run.sh --local                               # default ATU run
```

That's it. `run.sh` dispatches to every experiment mode in the repo.

---

## The one script: `run.sh`

```
bash run.sh --mode {train|chain|sweep|trace} [flags...]
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--mode` | all | `train` (default), `chain`, `sweep`, `trace` |
| `--target ID` | train/chain/sweep/trace | RWKU target, e.g. `1_Stephen_King` |
| `--layer N` | train/sweep | transformer hook layer (default 4) |
| `--task CFG` | train/chain | `unlearning_atu` (default), `unlearning_atu_align_only`, `unlearning_atu_single_rung`, `unlearning_ga`, `unlearning_npo` |
| `--experiment EXP` | train | Hydra experiment preset (`celebs-1`, `multi_turn_layers`, …) |
| `--model HF_ID` | trace | e.g. `Qwen/Qwen3.5-4B`, `microsoft/Phi-3.5-mini-instruct` |
| `--chain "t:L,t:L"` | chain | comma-separated `target:layer` pairs |
| `--widths "2 5 10"` | sweep | subject-mask window values to sweep |
| `--overrides "k=v k=v"` | train/chain | extra Hydra overrides |
| `--local` | all | run inline; skip SLURM `sbatch` |

Env overrides: `PROJECT_DIR`, `HF_HOME`, `VENV_ACTIVATE`, `BASE_DIR` (sweep output dir).

### Examples

```bash
# Default ATU run (Stephen King, layer 4), inline:
bash run.sh --local

# GA baseline on celebs-1, via SLURM:
bash run.sh --task unlearning_ga --experiment celebs-1

# Try layer 20 instead of the default 4:
bash run.sh --layer 20 --local

# Chained unlearning — forget three targets in sequence:
bash run.sh --mode chain \
    --chain "1_Stephen_King:4,2_Confucius:20,3_Elon_Musk:10" --local

# Mask-width sweep (reuses one alignment across widths):
bash run.sh --mode sweep --target 1_Stephen_King --layer 4 --widths "2 5 10"

# Causal trace (which layers store Stephen King's facts?):
bash run.sh --mode trace --target 1_Stephen_King --model Qwen/Qwen3.5-4B --local

# Arbitrary Hydra overrides:
bash run.sh --local --overrides "trainer.max_epochs=5 task.subject_mask_window=3"
```

---

## Repo layout

```
run.sh                    # unified entrypoint — start here
launch_training.py        # Hydra entry for train/chain modes
train.py                  # training loop (single + chained)
causal_trace.py           # ROME-style causal tracing
config/                   # Hydra configs (task, experiment, model, data)
project/                  # library code (data, tasks, eval, utils)
scripts/                  # SLURM-aware wrappers called by run.sh
  ├─ run_training.sh      # single train run
  ├─ run_chained_training.sh
  ├─ run_width_sweep.sh
  └─ run_causal_trace.sh
data/rwku/                # benchmark download + preprocessing
reports/                  # experiment writeups
```

---

## Method notes

### Align-then-Unlearn (baseline)
The LLM is first augmented with a small **embedding prediction module** trained to anticipate future-context embeddings. Unlearning then fine-tunes the LLM to drive those predicted embeddings *away from* a target concept's embedding, in the semantic space of a frozen sentence encoder. Because the signal lives in embedding space rather than on output tokens, it survives prompt rephrasing better than GA/NPO.

### What we added
- **Layer-informed hooking.** Instead of tapping the final layer, we choose the hook layer empirically via **causal tracing** (Meng et al. 2022). See `causal_trace.py` and `--mode trace`.
- **Subject-token masking.** The ReLU-cosine unlearning loss is applied only at positions that belong to the subject span (`task.subject_mask_window` dilates the span by N tokens on each side). Other positions are optionally anchored to the pre-unlearn model via KL-retain (`task.training_module.kl_retain_weight`).
- **Chained unlearning.** `--mode chain` forgets multiple (target, layer) pairs sequentially, feeding each step's modified model into the next, with pre/post USR/APR/GUR evals for every target.

### Causal tracing in one paragraph
For each prompt about a target (e.g. *"The Shining was written by"*):
1. **Clean pass** — record P(correct answer) and all hidden states.
2. **Corrupted pass** — add Gaussian noise (σ = `noise_multiplier` × embedding-layer std) to the subject token embeddings; answer probability collapses.
3. **Restore layer *l*** — run the corrupted input but patch layer *l* back to its clean hidden state. Measure recovery.

$$\text{Recovery}(l) = \frac{P^{(l)}_{\text{restored}} - P_{\text{corrupted}}}{P_{\text{clean}} - P_{\text{corrupted}}}$$

Layers with high average recovery across prompts are where the entity's knowledge lives — those are the layers we hook for unlearning.

---

## Requirements

- Python ≥ 3.10
- A CUDA GPU (A100 / H100 / H200 used in our runs; 24 GB works for Phi-3.5-mini)
- `transformers==4.51.3` is pinned — newer versions break the model hooks we use

See `pyproject.toml` and `requirements.txt` for the full list.

---

## Citation

```bibtex
@article{spohn2025align,
  title  = {Align-then-Unlearn: Embedding Alignment for LLM Unlearning},
  author = {Spohn, Philipp and Girrbach, Leander and Bader, Jessica and Akata, Zeynep},
  journal= {ICML 2025 Workshop on Machine Unlearning for Generative AI},
  year   = {2025}
}
```

## Acknowledgements
- Based on the template by [Marten Lienen](https://github.com/martenlienen).
- RWKU benchmark code adopted from [jinzhuoran/RWKU](https://github.com/jinzhuoran/RWKU).
- Original Align-then-Unlearn implementation from [ExplainableML/align-then-unlearn](https://github.com/ExplainableML/align-then-unlearn).
