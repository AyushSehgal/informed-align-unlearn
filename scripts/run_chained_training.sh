#!/bin/bash

# Chained Align-then-Unlearn Training Batch Submitter
# Runs multiple (target, layer) unlearning steps sequentially, passing the
# model modified by each step into the next.
#
# Usage:
#   bash run_chained_training.sh --chain "1_Stephen_King:4,2_Confucius:20"
#   bash run_chained_training.sh --chain "1_Stephen_King:4,2_Confucius:20" --task unlearning_ga
#   bash run_chained_training.sh --chain "1_Stephen_King:4,2_Confucius:20" --overrides "trainer.max_epochs=5"
#
# Chain format: comma-separated "target:layer" pairs
#   e.g. "1_Stephen_King:4,2_Confucius:20,3_Elon_Musk:10"

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

CHAIN=""
TASK=""
EXTRA_OVERRIDES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --chain)
            CHAIN="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --overrides)
            EXTRA_OVERRIDES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_chained_training.sh --chain \"target1:layer1,target2:layer2\" [--task TASK] [--overrides \"key=val ...\"]"
            exit 1
            ;;
    esac
done

if [ -z "$CHAIN" ]; then
    echo "Error: --chain is required"
    echo "Usage: bash run_chained_training.sh --chain \"target1:layer1,target2:layer2\""
    exit 1
fi

# ============================================================================
# BUILD HYDRA CHAIN OVERRIDE
# Convert "1_Stephen_King:4,2_Confucius:20" into the Hydra list override:
#   unlearning_chain="[{target: 1_Stephen_King, layer: 4},{target: 2_Confucius, layer: 20}]"
# ============================================================================

CHAIN_OVERRIDE=""
IFS=',' read -ra STEPS <<< "$CHAIN"
for STEP in "${STEPS[@]}"; do
    TARGET="${STEP%%:*}"
    LAYER="${STEP##*:}"
    if [ -z "$CHAIN_OVERRIDE" ]; then
        CHAIN_OVERRIDE="{target: ${TARGET}, layer: ${LAYER}}"
    else
        CHAIN_OVERRIDE="${CHAIN_OVERRIDE},{target: ${TARGET}, layer: ${LAYER}}"
    fi
done
CHAIN_OVERRIDE="unlearning_chain=[${CHAIN_OVERRIDE}]"

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR="/data/user_data/ayushseh/informed-align-unlearn"
LOG_DIR="${PROJECT_DIR}/logs/unlearn"

# Build a descriptive job suffix from the chain
CHAIN_SLUG=$(echo "$CHAIN" | tr ',' '-' | tr ':' 'L')
JOB_SUFFIX="chain_${CHAIN_SLUG}"
if [ -n "$TASK" ]; then
    JOB_SUFFIX="${JOB_SUFFIX}_${TASK}"
fi

mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING CHAINED UNLEARNING JOB"
echo "========================================="
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Chain: ${CHAIN}"
echo "Task: ${TASK:-default}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-none}"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# BUILD HYDRA COMMAND
# ============================================================================

HYDRA_CMD="python3 launch_training.py \"${CHAIN_OVERRIDE}\""

if [ -n "$TASK" ]; then
    HYDRA_CMD="${HYDRA_CMD} task=${TASK}"
fi

if [ -n "$EXTRA_OVERRIDES" ]; then
    HYDRA_CMD="${HYDRA_CMD} ${EXTRA_OVERRIDES}"
fi

echo "Hydra command: ${HYDRA_CMD}"
echo ""

# ============================================================================
# SUBMIT JOB
# ============================================================================

JOB_ID=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=chain_${JOB_SUFFIX}
#SBATCH --output=${LOG_DIR}/${JOB_SUFFIX}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_SUFFIX}_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
source idl/venv/bin/activate

echo "Running: ${HYDRA_CMD}"
echo ""

${HYDRA_CMD}

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
echo "View stdout:     tail -f ${LOG_DIR}/${JOB_SUFFIX}_${JOB_NUM}.out"
echo "View stderr:     tail -f ${LOG_DIR}/${JOB_SUFFIX}_${JOB_NUM}.err"
echo "Cancel with:     scancel ${JOB_NUM}"
echo ""
