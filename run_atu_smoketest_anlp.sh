#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 04:00:00
#SBATCH -A cis250260p
#SBATCH --job-name=atu_smoketest
#SBATCH --output=output_smoketest_%j.log
#SBATCH --error=error_smoketest_%j.log

# ==============================================================================
# Multi-Layer ATU Smoke Test (V100-32, minimal steps)
# ==============================================================================
# Quick run to catch code errors before submitting a full H100 job.
# Overrides stages to just 5 training + 5 unlearning steps total.
# Uses V100-32 (32GB) — memory is tight during unlearning but should fit.
# ==============================================================================

set -e

# --- Environment setup ---
# Use pure-Python protobuf to avoid GLIBCXX_3.4.29 not found on compute nodes
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
module load anaconda3
conda activate informed-align-unlearn
cd /jet/home/apatawar/informed-align-unlearn
source .env

export HF_HUB_CACHE="/ocean/projects/cis250260p/shared/hf_cache"

UNLEARNING_TARGET="1_Stephen_King"

echo "============================================================"
echo "Multi-Layer ATU Smoke Test"
echo "Target: $UNLEARNING_TARGET"
echo "GPU: V100-32 (1x)"
echo "============================================================"

python launch_training.py \
    unlearning_target="$UNLEARNING_TARGET" \
    'task.training_module.pretrained_model_hook_layers=[4,15,29]' \
    task.first_stage_steps=5 \
    'task.stages=[{type: training, steps: 5},{type: unlearning, threshold: 0.9, steps: 5}]' \
    skip_initial_eval=true \
    wandb.mode=disabled \
    wandb.name=atu-smoketest

echo ""
echo "============================================================"
echo "Smoke test complete! No errors found."
echo "============================================================"
