#!/bin/bash

# Causal Tracing Batch Submitter
# Submits sbatch job for causal tracing analysis
#
# Usage:
#   bash run_causal_trace.sh                                          # Default (1_Stephen_King, Phi-3.5-mini)
#   bash run_causal_trace.sh --target_id 9_Justin_Bieber              # Different target
#   bash run_causal_trace.sh --model microsoft/Phi-3.5-mini-instruct  # Different model
#   bash run_causal_trace.sh --top_k 10 --noise_multiplier 5.0        # Custom params

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

TARGET_ID="1_Stephen_King"
MODEL="microsoft/Phi-3.5-mini-instruct"
TOP_K="5"
NOISE_MULTIPLIER="3.0"
DTYPE="bfloat16"

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
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_causal_trace.sh [--target_id ID] [--model MODEL] [--top_k K] [--noise_multiplier N] [--dtype DTYPE]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR="/data/user_data/ayushseh/informed-align-unlearn"
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

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd ${PROJECT_DIR}
source venv/bin/activate

python3 causal_trace.py \
    --model "${MODEL}" \
    --target_id "${TARGET_ID}" \
    --top_k ${TOP_K} \
    --noise_multiplier ${NOISE_MULTIPLIER} \
    --dtype ${DTYPE}

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
