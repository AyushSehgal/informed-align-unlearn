#!/bin/bash
# LUNE Baseline Experiment — sbatch launcher
#
# Usage:
#   bash run_lune.sh              # submit the job
#   bash run_lune.sh --dry-run    # print the sbatch script without submitting

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

LOG_DIR="logs/lune"
mkdir -p "$LOG_DIR"

SBATCH_SCRIPT=$(cat <<'HEREDOC'
#!/bin/bash
#SBATCH --job-name=lune_phi35mini
#SBATCH --output=logs/lune/%j.out
#SBATCH --error=logs/lune/%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=preempt

echo "============================================"
echo "LUNE Baseline — Phi-3.5-mini-instruct on RWKU"
echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start:     $(date)"
echo ""

export TRANSFORMERS_NO_TF=1

python lune_train.py

echo ""
echo "End: $(date)"
HEREDOC
)

if $DRY_RUN; then
    echo "$SBATCH_SCRIPT"
    echo ""
    echo "(dry run — not submitted)"
else
    JOB_ID=$(echo "$SBATCH_SCRIPT" | sbatch)
    echo "Submitted: $JOB_ID"
    echo "Logs:      $LOG_DIR/"
    echo "Monitor:   squeue -u \$USER"
fi
