#!/bin/bash
# Mask-width sweep at a fixed layer (default 4). Runs alignment once, then
# cycles through subject_mask_window values, loading the shared aligned
# checkpoint each time so the LLM starts fresh at every sweep point.
#
# Usage (on the pod):
#   bash scripts/run_width_sweep.sh
#
# Env overrides:
#   TARGET=1_Stephen_King  HOOK_LAYER=4  WIDTHS="2 5 10"  BASE_DIR=/workspace/runs/sweep
set -euo pipefail

: "${TARGET:=1_Stephen_King}"
: "${HOOK_LAYER:=4}"
: "${WIDTHS:=2 5 10}"
: "${BASE_DIR:=/workspace/runs/sweep}"
: "${HF_HOME:=/workspace/.cache/huggingface}"

export HF_HOME
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

ALIGN_DIR="${BASE_DIR}/align_only"
mkdir -p "${BASE_DIR}"

# ---- Step 1: align once (skip if already done) ------------------------------
if [ ! -f "${ALIGN_DIR}/ckpt_path.txt" ]; then
    echo "=== [sweep] Aligning once at layer=${HOOK_LAYER} ==="
    python3 launch_training.py \
        task=unlearning_atu_align_only \
        unlearning_target="${TARGET}" \
        skip_initial_eval=true \
        task.training_module.pretrained_model_hook_layer="${HOOK_LAYER}" \
        task.unlearning_data.batch_size=32 \
        task.unlearning_data.max_input_length=128 \
        task.unlearning_data.num_workers=8 \
        trainer.devices=1 \
        +trainer.precision=bf16-mixed \
        hydra.run.dir="${ALIGN_DIR}"

    # align-only is a single training stage, so the save path writes
    # unlearned_*.pt instead of pre_trained_llm.pt. stage1_checkpoint
    # loading expects the plain filenames — copy them over.
    CKPT_DIR="$(find "${ALIGN_DIR}/checkpoints" -type d -name "${TARGET}" | head -1)"
    cp "${CKPT_DIR}/unlearned_pre_trained_llm.pt" "${CKPT_DIR}/pre_trained_llm.pt"
    cp "${CKPT_DIR}/unlearned_embedding_prediction_model.pt" \
       "${CKPT_DIR}/embedding_prediction_model.pt"
    echo "${CKPT_DIR}" > "${ALIGN_DIR}/ckpt_path.txt"
    echo "=== [sweep] alignment saved at ${CKPT_DIR} ==="
else
    echo "=== [sweep] reusing alignment from ${ALIGN_DIR} ==="
fi
CKPT_DIR="$(cat "${ALIGN_DIR}/ckpt_path.txt")"

# ---- Step 2: sweep ----------------------------------------------------------
for W in ${WIDTHS}; do
    RUN_DIR="${BASE_DIR}/w${W}"
    if [ -f "${RUN_DIR}/done.txt" ]; then
        echo "=== [sweep] width=${W} already done, skipping ==="
        continue
    fi
    echo "=== [sweep] width=${W} ==="
    python3 launch_training.py \
        task=unlearning_atu_single_rung \
        unlearning_target="${TARGET}" \
        skip_initial_eval=true \
        task.stage1_checkpoint="${CKPT_DIR}" \
        task.training_module.pretrained_model_hook_layer="${HOOK_LAYER}" \
        task.training_module.kl_retain_weight=0.0 \
        task.training_module.disable_grad_checkpointing_on_unlearn=true \
        task.subject_mask_window="${W}" \
        task.unlearning_data.batch_size=32 \
        task.unlearning_data.max_input_length=128 \
        task.unlearning_data.num_workers=8 \
        trainer.devices=1 \
        +trainer.precision=bf16-mixed \
        hydra.run.dir="${RUN_DIR}"
    touch "${RUN_DIR}/done.txt"
done

# ---- Step 3: print summary --------------------------------------------------
echo ""
echo "=========================================="
echo "SWEEP SUMMARY (layer=${HOOK_LAYER}, target=${TARGET})"
echo "=========================================="
for W in ${WIDTHS}; do
    RUN_DIR="${BASE_DIR}/w${W}"
    echo ""
    echo "--- width=${W} ---"
    if [ -f "${RUN_DIR}/atu-unlearning.log" ]; then
        grep -E "USR|APR|utility/gen|utility/flu|forget/fb|forget/qa|neighbor/fb" \
             "${RUN_DIR}/atu-unlearning.log" | tail -20 || true
    else
        echo "no log at ${RUN_DIR}"
    fi
done
