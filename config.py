"""Central configuration for the unlearning pipeline."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Allow overriding output dir via environment variable (for Colab + Google Drive)
_output_dir = os.environ.get("OUTPUT_DIR")
SAVES_DIR = Path(_output_dir) / "saves" if _output_dir else ROOT / "saves"

REINFORCED_DIR = SAVES_DIR / "reinforced"
UNLEARNED_DIR = SAVES_DIR / "unlearned"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_NAME = "Stephen King"
RWKU_DATASET = "jinzhuoran/RWKU"

# ── Reinforced model training ─────────────────────────────────────────────────
REINFORCED_LR = 3e-6
REINFORCED_EPOCHS = 3
REINFORCED_BATCH = 1
REINFORCED_GRAD_ACCUM = 8  # effective batch = 8
REINFORCED_CTX_LEN = 512

# ── Unlearning training ───────────────────────────────────────────────────────
UNLEARN_LR = 1e-6
UNLEARN_EPOCHS = 2
UNLEARN_BATCH = 1
UNLEARN_GRAD_ACCUM = 8  # effective batch = 8

# ── Logit blending ────────────────────────────────────────────────────────────
ALPHA = 5.0
