#!/bin/bash

# Causal Tracing Batch Submitter
# Submits an sbatch job for causal-tracing analysis, OR runs locally with
# --local if SLURM isn't available.
#
# Usage:
#   bash run_causal_trace.sh                                          # Default (1_Stephen_King, Qwen3.5-4B)
#   bash run_causal_trace.sh --target_id 9_Justin_Bieber              # Different target
#   bash run_causal_trace.sh --model microsoft/Phi-3.5-mini-instruct  # Different model
#   bash run_causal_trace.sh --top_k 10 --noise_multiplier 5.0        # Custom params
#   bash run_causal_trace.sh --local                                  # Run inline, no sbatch
#
# Environment overrides:
#   PROJECT_DIR   absolute path to the repo checkout (defaults to git root)
#   HF_HOME       cache directory for HuggingFace downloads
#   VENV_ACTIVATE path to a venv activate script to source

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

TARGET_ID="1_Stephen_King"
MODEL="Qwen/Qwen3.5-4B"
TOP_K="5"
NOISE_MULTIPLIER="3.0"
DTYPE="bfloat16"
LOCAL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_id)
            TARGET_ID="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --noise_multiplier)
            NOISE_MULTIPLIER="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --local)
            LOCAL=1
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_causal_trace.sh [--target_id ID] [--model MODEL] [--top_k K] [--noise_multiplier N] [--dtype DTYPE] [--local]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default PROJECT_DIR to the git root; override via env if needed.
PROJECT_DIR="${PROJECT_DIR:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || pwd)}"
LOG_DIR="${PROJECT_DIR}/logs/causal_trace"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING CAUSAL TRACING JOB"
echo "========================================="
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Target ID: $TARGET_ID"
echo "Top-k: $TOP_K"
echo "Noise multiplier: $NOISE_MULTIPLIER"
echo "Dtype: $DTYPE"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# SUBMIT JOB
# ============================================================================

# Auto-enable trust_remote_code for Qwen (model card requires it on some
# variants). Harmless for Phi models because we don't pass the flag at all.
EXTRA_ARGS=""
case "${MODEL}" in
    *[Qq]wen*) EXTRA_ARGS="--trust_remote_code --device_map auto" ;;
esac

run_trace() {
    cd "${PROJECT_DIR}"
    if [ -n "${VENV_ACTIVATE:-}" ] && [ -f "${VENV_ACTIVATE}" ]; then
        # shellcheck disable=SC1090
        source "${VENV_ACTIVATE}"
    fi
    python3 causal_trace.py \
        --model "${MODEL}" \
        --target_id "${TARGET_ID}" \
        --top_k "${TOP_K}" \
        --noise_multiplier "${NOISE_MULTIPLIER}" \
        --dtype "${DTYPE}" \
        ${EXTRA_ARGS}
}

if [ "${LOCAL}" = "1" ] || ! command -v sbatch > /dev/null 2>&1; then
    echo "Running locally (no sbatch)"
    run_trace
    exit $?
fi

JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=causal_trace_${TARGET_ID}
#SBATCH --output=${LOG_DIR}/${TARGET_ID}_%j.out
#SBATCH --error=${LOG_DIR}/${TARGET_ID}_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --partition=general

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export HF_HOME=${HF_HOME:-\$HOME/.cache/huggingface}

cd ${PROJECT_DIR}
if [ -n "${VENV_ACTIVATE:-}" ] && [ -f "${VENV_ACTIVATE}" ]; then
    source "${VENV_ACTIVATE}"
fi

python3 causal_trace.py \\
    --model "${MODEL}" \\
    --target_id "${TARGET_ID}" \\
    --top_k ${TOP_K} \\
    --noise_multiplier ${NOISE_MULTIPLIER} \\
    --dtype ${DTYPE} ${EXTRA_ARGS}

echo ""
echo "End time: \$(date)"
EOF
)

JOB_NUM="${JOB_ID##* }"

echo "  Submitted: $JOB_ID"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================="
echo "JOB SUBMITTED!"
echo "========================================="
echo ""
echo "Monitor with:    squeue -u \$USER"
echo "View stdout:     tail -f ${LOG_DIR}/${TARGET_ID}_${JOB_NUM}.out"
echo "View stderr:     tail -f ${LOG_DIR}/${TARGET_ID}_${JOB_NUM}.err"
echo "Cancel with:     scancel ${JOB_NUM}"
echo ""
