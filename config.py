"""Central configuration for the unlearning pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAVES_DIR = ROOT / "saves"
RWKU_DIR = DATA_DIR / "RWKU"
TARGET_DIR = RWKU_DIR / "Target" / "141_J._K._Rowling"

REINFORCED_DIR = SAVES_DIR / "reinforced"
UNLEARNED_DIR = SAVES_DIR / "unlearned"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_NAME = "J. K. Rowling"
TARGET_ID = "141_J._K._Rowling"

# ── Reinforced model training ─────────────────────────────────────────────────
REINFORCED_LR = 3e-6
REINFORCED_EPOCHS = 3
REINFORCED_BATCH = 1
REINFORCED_GRAD_ACCUM = 128  # effective batch = 128
REINFORCED_CTX_LEN = 512

# ── Unlearning training ───────────────────────────────────────────────────────
UNLEARN_LR = 1e-6
UNLEARN_EPOCHS = 2
UNLEARN_BATCH = 1
UNLEARN_GRAD_ACCUM = 128  # effective batch = 128

# ── Logit blending ────────────────────────────────────────────────────────────
ALPHA = 5.0

# ── RWKU evaluation ───────────────────────────────────────────────────────────
RWKU_GDRIVE_ID = "1SKbZ8SNJtMrfVVlwkAS3DTMqRkIGlodc"
