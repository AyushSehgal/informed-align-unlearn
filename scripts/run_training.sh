#!/bin/bash

# Align-then-Unlearn Training Batch Submitter
# Submits sbatch job(s) for the align-then-unlearn project (Hydra-based)
#
# Usage:
#   bash run_unlearn.sh                                    # Run default task with default config
#   bash run_unlearn.sh --task unlearning_ga                # Run with GA unlearning task
#   bash run_unlearn.sh --experiment celebs-1               # Run with celebs-1 experiment config
#   bash run_unlearn.sh --overrides "training.lr=1e-4"      # Pass extra Hydra overrides

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

TASK=""
EXPERIMENT=""
EXTRA_OVERRIDES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --overrides)
            EXTRA_OVERRIDES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_unlearn.sh [--task TASK] [--experiment EXPERIMENT] [--overrides \"key=val ...\"]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR="/data/user_data/ayushseh/informed-align-unlearn"
LOG_DIR="${PROJECT_DIR}/logs/unlearn"

# Build a descriptive job suffix
JOB_SUFFIX="default"
if [ -n "$TASK" ]; then
    JOB_SUFFIX="${TASK}"
fi
if [ -n "$EXPERIMENT" ]; then
    JOB_SUFFIX="${JOB_SUFFIX}_${EXPERIMENT}"
fi

# Create logs directory
mkdir -p "$LOG_DIR"

echo "========================================="
echo "SUBMITTING ALIGN-THEN-UNLEARN JOB"
echo "========================================="
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Task: ${TASK:-default}"
echo "Experiment: ${EXPERIMENT:-default}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-none}"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# BUILD HYDRA COMMAND
# ============================================================================

HYDRA_CMD="python3 launch_training.py task.training_module.pretrained_model_hook_layer=4"

if [ -n "$TASK" ]; then
    HYDRA_CMD="${HYDRA_CMD} task=${TASK}"
fi

if [ -n "$EXPERIMENT" ]; then
    HYDRA_CMD="${HYDRA_CMD} experiment=${EXPERIMENT}"
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
#SBATCH --job-name=unlearn_${JOB_SUFFIX}
#SBATCH --output=${LOG_DIR}/${JOB_SUFFIX}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_SUFFIX}_%j.err
#SBATCH --time=24:00:00
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
source venv/bin/activate

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