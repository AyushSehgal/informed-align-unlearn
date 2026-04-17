#!/bin/bash
#SBATCH --job-name=atu_multi_layer
#SBATCH --output=/data/user_data/ayushseh/informed-align-unlearn/logs/unlearn/multi-layer_%j.out
#SBATCH --error=/data/user_data/ayushseh/informed-align-unlearn/logs/unlearn/multi-layer_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=general

PROJECT_DIR="/data/user_data/ayushseh/informed-align-unlearn"
LOG_DIR="${PROJECT_DIR}/logs/unlearn"
JOB_SUFFIX="multi-layer"

# ==============================================================================
# Multi-Layer ATU Training Script
# ==============================================================================
# Runs the multi-layer Align-Then-Unlearn (ATU) training experiments with
# independent prediction modules attached at multiple transformer layers.
#
# Three configurations (edit/comment-out as needed):
#   - atu-multi-4-15-29 : early + mid + late layers (default config)
#   - atu-multi-4-29    : causal peak + ATU default
#   - atu-single-29     : single-layer baseline (layer 29 only)
#
# Each run uses the target specified by UNLEARNING_TARGET below.
#
# Usage:
#   sbatch run_atu_multi_layer.sh
#
# Or run a specific experiment by commenting out the others.
# ==============================================================================

set -e

# --- Environment setup ---
# Use pure-Python protobuf to avoid GLIBCXX_3.4.29 not found on compute nodes
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export HF_HOME=/data/user_data/ayushseh/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd /data/user_data/ayushseh/informed-align-unlearn
source idl/venv/bin/activate

UNLEARNING_TARGET="1_Stephen_King"

echo "============================================================"
echo "Multi-Layer ATU Training"
echo "Target: $UNLEARNING_TARGET"
echo "============================================================"

# ==============================================================================
# 1. Three layers: early + mid + late (default config)
# ==============================================================================
echo ""
echo ">>> [1/3] Three-layer ATU: layers 4, 15, 29"
echo "------------------------------------------------------------"

python launch_training.py \
    unlearning_target="$UNLEARNING_TARGET" \
    'task.training_module.pretrained_model_hook_layers=[4,15,29]' \
    wandb.name=atu-multi-4-15-29

# ==============================================================================
# 2. Two layers: causal peak + ATU default
# ==============================================================================
# echo ""
# echo ">>> [2/3] Two-layer ATU: layers 4, 29"
# echo "------------------------------------------------------------"

# python launch_training.py \
#     unlearning_target="$UNLEARNING_TARGET" \
#     'task.training_module.pretrained_model_hook_layers=[4,29]' \
#     wandb.name=atu-multi-4-29

# ==============================================================================
# 3. Single-layer baseline (layer 29 only)
# ==============================================================================
# echo ""
# echo ">>> [3/3] Single-layer ATU baseline: layer 29"
# echo "------------------------------------------------------------"

# python launch_training.py \
#     unlearning_target="$UNLEARNING_TARGET" \
#     'task.training_module.pretrained_model_hook_layers=[29]' \
#     wandb.name=atu-single-29

echo ""
echo "============================================================"
echo "ATU training complete!"
echo "============================================================"
