#!/bin/bash
# Unified entrypoint for the Informed Align-then-Unlearn project.
#
# One script, one --mode flag, every experiment. Delegates to the specialized
# scripts under scripts/ for SLURM-aware submission.
#
# ----------------------------------------------------------------------------
# MODES
# ----------------------------------------------------------------------------
#   train  — single align-then-unlearn run (default)
#   chain  — sequential unlearning of multiple (target, layer) pairs
#   sweep  — mask-width sweep at a fixed layer (reuses a single alignment)
#   trace  — ROME-style causal tracing (which layers store the target's facts?)
#
# Any --mode can also be --local to skip SLURM and run inline.
#
# ----------------------------------------------------------------------------
# COMMON FLAGS
# ----------------------------------------------------------------------------
#   --mode {train|chain|sweep|trace}       what to run (default: train)
#   --target TARGET_ID                     e.g. 1_Stephen_King (RWKU id)
#   --layer N                              transformer hook layer (default 4)
#   --task TASK_CONFIG                     e.g. unlearning_atu (default),
#                                          unlearning_atu_align_only,
#                                          unlearning_atu_single_rung,
#                                          unlearning_ga, unlearning_npo
#   --experiment EXP                       Hydra experiment preset
#                                          (e.g. celebs-1, multi_turn_layers)
#   --model MODEL                          HF model id (trace mode only)
#   --chain "t1:L1,t2:L2,..."              chain targets (chain mode)
#   --widths "2 5 10"                      subject_mask_window values (sweep)
#   --overrides "k1=v1 k2=v2"              extra Hydra overrides
#   --local                                run inline, skip sbatch
#   --help | -h                            this message
#
# ----------------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------------
#   # Default ATU run, Stephen King, layer 4, local:
#   bash run.sh --local
#
#   # GA baseline on celebs-1, on SLURM:
#   bash run.sh --task unlearning_ga --experiment celebs-1
#
#   # Chained unlearning of 3 targets, local:
#   bash run.sh --mode chain --chain "1_Stephen_King:4,2_Confucius:20,3_Elon_Musk:10" --local
#
#   # Mask-width sweep at layer 4:
#   bash run.sh --mode sweep --target 1_Stephen_King --layer 4 --widths "2 5 10"
#
#   # Causal trace on Qwen3.5-4B for Stephen King:
#   bash run.sh --mode trace --target 1_Stephen_King --model Qwen/Qwen3.5-4B --local
#
# ----------------------------------------------------------------------------
# ENV OVERRIDES
# ----------------------------------------------------------------------------
#   PROJECT_DIR    absolute path to repo (defaults to git root)
#   HF_HOME        HF cache dir (default ~/.cache/huggingface)
#   VENV_ACTIVATE  path to a venv activate script to source
#   BASE_DIR       sweep-mode output dir (default /workspace/runs/sweep)
# ============================================================================

set -euo pipefail

MODE="train"
TARGET=""
LAYER=""
TASK=""
EXPERIMENT=""
MODEL=""
CHAIN=""
WIDTHS=""
OVERRIDES=""
LOCAL_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2"; shift 2 ;;
        --target)      TARGET="$2"; shift 2 ;;
        --target_id)   TARGET="$2"; shift 2 ;;
        --layer)       LAYER="$2"; shift 2 ;;
        --task)        TASK="$2"; shift 2 ;;
        --experiment)  EXPERIMENT="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --chain)       CHAIN="$2"; shift 2 ;;
        --widths)      WIDTHS="$2"; shift 2 ;;
        --overrides)   OVERRIDES="$2"; shift 2 ;;
        --local)       LOCAL_FLAG="--local"; shift 1 ;;
        -h|--help)
            sed -n '2,60p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run 'bash run.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

case "$MODE" in
    train)
        EXTRA="${OVERRIDES}"
        if [ -n "$TARGET" ]; then
            EXTRA="unlearning_target=${TARGET} ${EXTRA}"
        fi
        if [ -n "$LAYER" ]; then
            EXTRA="task.training_module.pretrained_model_hook_layer=${LAYER} ${EXTRA}"
        fi
        ARGS=()
        [ -n "$TASK" ]       && ARGS+=(--task "$TASK")
        [ -n "$EXPERIMENT" ] && ARGS+=(--experiment "$EXPERIMENT")
        [ -n "$EXTRA" ]      && ARGS+=(--overrides "$EXTRA")
        [ -n "$LOCAL_FLAG" ] && ARGS+=("$LOCAL_FLAG")
        exec bash "${SCRIPTS_DIR}/run_training.sh" "${ARGS[@]}"
        ;;

    chain)
        if [ -z "$CHAIN" ]; then
            echo "--mode chain requires --chain \"t1:L1,t2:L2,...\"" >&2
            exit 1
        fi
        ARGS=(--chain "$CHAIN")
        [ -n "$TASK" ]       && ARGS+=(--task "$TASK")
        [ -n "$OVERRIDES" ]  && ARGS+=(--overrides "$OVERRIDES")
        [ -n "$LOCAL_FLAG" ] && ARGS+=("$LOCAL_FLAG")
        exec bash "${SCRIPTS_DIR}/run_chained_training.sh" "${ARGS[@]}"
        ;;

    sweep)
        [ -n "$TARGET" ]  && export TARGET
        [ -n "$LAYER" ]   && export HOOK_LAYER="$LAYER"
        [ -n "$WIDTHS" ]  && export WIDTHS
        exec bash "${SCRIPTS_DIR}/run_width_sweep.sh"
        ;;

    trace)
        ARGS=()
        [ -n "$TARGET" ]     && ARGS+=(--target_id "$TARGET")
        [ -n "$MODEL" ]      && ARGS+=(--model "$MODEL")
        [ -n "$LOCAL_FLAG" ] && ARGS+=("$LOCAL_FLAG")
        exec bash "${SCRIPTS_DIR}/run_causal_trace.sh" "${ARGS[@]}"
        ;;

    *)
        echo "Unknown --mode: $MODE (expected: train, chain, sweep, trace)" >&2
        exit 1
        ;;
esac
