#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 12:00:00
#SBATCH -A cis250260p
#SBATCH --job-name=atu_ablation_15_29
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# ==============================================================================
# Ablation: Two-layer ATU at layers 15 and 29
# ==============================================================================

set -e

# --- Environment setup ---
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
module load anaconda3
conda activate informed-align-unlearn
cd /jet/home/apatawar/informed-align-unlearn
source .env

export HF_HUB_CACHE="/ocean/projects/cis250260p/shared/hf_cache"

UNLEARNING_TARGET="1_Stephen_King"
SEED=257640305935986723276204284023905265734

echo "============================================================"
echo "Ablation: Two-layer ATU at layers 15 and 29"
echo "Target: $UNLEARNING_TARGET"
echo "Seed:   $SEED"
echo "============================================================"

python launch_training.py \
    unlearning_target="$UNLEARNING_TARGET" \
    'task.training_module.pretrained_model_hook_layers=[15,29]' \
    seed="$SEED" \
    wandb.name=atu-ablation-15-29

echo ""
echo "============================================================"
echo "Ablation complete!"
echo "============================================================"
