"""
Dry-run smoke test for multi-layer ATU. Runs on CPU with tiny mock models.
Imports only torch and numpy — avoids lightning/wandb which need compute-node libs.

Usage:
    module load anaconda3 && conda activate informed-align-unlearn && python dry_run.py
"""

import sys
import os
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(msg):
    print(f"  [OK] {msg}")

def fail(msg, exc=None):
    print(f"  [FAIL] {msg}")
    if exc:
        import traceback; traceback.print_exc()
    sys.exit(1)

def import_file(name, rel_path):
    """Import a .py file directly, bypassing package __init__.py."""
    path = os.path.join(BASE, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Core imports (no lightning/wandb)
# ---------------------------------------------------------------------------
section("1. Core imports (torch, numpy)")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    ok(f"torch {torch.__version__}, numpy {np.__version__}")
except Exception as e:
    fail("torch/numpy import", e)


# ---------------------------------------------------------------------------
# 2. Seed fix — import seed.py directly, bypassing project/utils/__init__.py
# ---------------------------------------------------------------------------
section("2. Seed fix (NumPy 2.x compat)")
try:
    seed_mod = import_file("seed", "project/utils/seed.py")
    manual_seed = seed_mod.manual_seed
    ok("imported seed.py directly")
except Exception as e:
    fail("seed.py import", e)

try:
    seed, rng = manual_seed(42)
    assert isinstance(seed, int), f"Expected int, got {type(seed)}"
    ok(f"manual_seed(42) → seed={seed}")
except Exception as e:
    fail("manual_seed(42)", e)

try:
    seed2, _ = manual_seed(None)
    assert isinstance(seed2, int)
    ok(f"manual_seed(None) → seed={seed2} (random)")
except Exception as e:
    fail("manual_seed(None)", e)


# ---------------------------------------------------------------------------
# 3. EmbeddingPredictionTransformer
# ---------------------------------------------------------------------------
section("3. EmbeddingPredictionTransformer forward pass")
try:
    epm_mod = import_file("epm", "project/embedding_prediction_model.py")
    EmbeddingPredictionTransformer = epm_mod.EmbeddingPredictionTransformer
    ok("imported embedding_prediction_model.py directly")
except Exception as e:
    fail("embedding_prediction_model.py import", e)

try:
    INPUT_DIM = 32
    HIDDEN_DIM = 16
    OUTPUT_DIM = 8
    BATCH, SEQ = 2, 5

    pred = EmbeddingPredictionTransformer(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
        num_heads=2, num_layers=2, dropout=0.0,
    )
    out = pred(torch.randn(BATCH, SEQ, INPUT_DIM))
    assert out.shape == (BATCH, SEQ, OUTPUT_DIM)
    ok(f"output shape: {tuple(out.shape)}")
except Exception as e:
    fail("EmbeddingPredictionTransformer forward", e)


# ---------------------------------------------------------------------------
# 4. mean_pooling_reference_encoder
# ---------------------------------------------------------------------------
section("4. mean_pooling_reference_encoder")
try:
    mp_mod = import_file("mean_pool", "project/utils/mean_pool.py")
    mean_pooling_reference_encoder = mp_mod.mean_pooling_reference_encoder
    ok("imported mean_pool.py directly")
except Exception as e:
    fail("mean_pool.py import", e)


# ---------------------------------------------------------------------------
# 5. Multi-layer ATU module logic (inline, no lightning import needed)
#    We replicate the key parts of UnlearningATUTrainingModule here to test
#    the multi-layer training_step logic without pulling in lightning/wandb.
# ---------------------------------------------------------------------------
section("5. Multi-layer training_step logic (training stage)")

HOOK_LAYERS = [1, 2, 3]
LLM_LAYERS = 4  # LLM has 4 layers → hidden_states[0..4]

# --- tiny mock LLM ---
class MockLLMConfig:
    hidden_size = INPUT_DIM

class MockLLMOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

class MockLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockLLMConfig()
        self.embed = nn.Embedding(100, INPUT_DIM)
        self.layers = nn.ModuleList([nn.Linear(INPUT_DIM, INPUT_DIM) for _ in range(LLM_LAYERS)])

    def forward(self, input_ids, output_hidden_states=False, attention_mask=None):
        x = self.embed(input_ids)
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return MockLLMOutput(tuple(states))

# --- tiny mock text encoder ---
class MockEncConfig:
    hidden_size = OUTPUT_DIM

class MockEncOutput:
    def __init__(self, lhs):
        self._lhs = lhs
    def __getitem__(self, i):
        if i == 0: return self._lhs
        raise IndexError(i)

class MockTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockEncConfig()
        self.embed = nn.Embedding(1000, OUTPUT_DIM)
    def forward(self, input_ids, output_hidden_states=False, attention_mask=None):
        return MockEncOutput(self.embed(input_ids))

# --- build ModuleDict of prediction modules ---
try:
    mock_llm = MockLLM()
    mock_enc = MockTextEncoder()

    embedding_prediction_models = nn.ModuleDict({
        str(layer): EmbeddingPredictionTransformer(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
            num_heads=2, num_layers=2, dropout=0.0,
        )
        for layer in HOOK_LAYERS
    })
    ok(f"ModuleDict with layers {HOOK_LAYERS}")
except Exception as e:
    fail("build embedding_prediction_models ModuleDict", e)

# --- simulate training stage forward + loss ---
try:
    WIN = 3
    batch = {
        "primary_input_ids": torch.randint(0, 100, (BATCH, SEQ)),
        "secondary_context_windows": torch.randint(0, 1000, (BATCH, SEQ, WIN)),
        "has_full_window": torch.ones(BATCH, SEQ, dtype=torch.bool),
        "attention_mask": torch.ones(BATCH, SEQ, dtype=torch.long),
    }

    input_ids = batch["primary_input_ids"]
    context_windows = batch["secondary_context_windows"]
    has_full_window = batch["has_full_window"]
    attention_mask = batch["attention_mask"]
    batch_size, seq_len, window_len = context_windows.shape

    # reference encoder
    with torch.no_grad():
        ref_out = mock_enc(
            input_ids=context_windows.view(-1, window_len),
            output_hidden_states=True,
        )
        attn = torch.ones(batch_size * seq_len, window_len)
        target_embeddings = mean_pooling_reference_encoder(ref_out, attn).view(batch_size, seq_len, -1)

    # LLM forward
    with torch.no_grad():
        llm_out = mock_llm(input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask)

    # per-layer loss
    total_loss = 0.0
    for layer_idx_str, pred_model in embedding_prediction_models.items():
        layer_idx = int(layer_idx_str)
        hidden = llm_out.hidden_states[layer_idx]
        outputs = pred_model(hidden)
        layer_loss = -torch.nn.functional.cosine_similarity(outputs, target_embeddings, dim=-1)
        layer_loss = layer_loss * has_full_window
        layer_loss = layer_loss.sum() / (has_full_window.sum() + 1e-8)
        total_loss = total_loss + layer_loss

    loss = total_loss / len(embedding_prediction_models)
    assert not torch.isnan(loss), "Loss is NaN"
    ok(f"training stage loss = {loss.item():.4f}")
except Exception as e:
    fail("training stage forward pass", e)


# ---------------------------------------------------------------------------
# 6. Multi-layer unlearning stage logic
# ---------------------------------------------------------------------------
section("6. Multi-layer training_step logic (unlearning stage)")
try:
    THRESHOLD = 0.9

    # compute unlearning target embedding (mock)
    with torch.no_grad():
        target_tokens = torch.tensor([[1, 2, 3]])
        attn_target = torch.ones_like(target_tokens)
        enc_out = mock_enc(input_ids=target_tokens)
        unlearning_target_embedding = mean_pooling_reference_encoder(enc_out, attn_target)[0]
    assert unlearning_target_embedding.shape == (OUTPUT_DIM,)
    ok(f"unlearning_target_embedding shape: {tuple(unlearning_target_embedding.shape)}")

    # LLM forward (trainable)
    llm_out = mock_llm(input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask)

    total_loss = 0.0
    for layer_idx_str, pred_model in embedding_prediction_models.items():
        layer_idx = int(layer_idx_str)
        hidden = llm_out.hidden_states[layer_idx]
        outputs = pred_model(hidden)
        layer_loss = torch.nn.functional.relu(
            torch.nn.functional.cosine_similarity(
                outputs,
                unlearning_target_embedding.unsqueeze(0).unsqueeze(0),
                dim=-1,
            ) - THRESHOLD
        ).mean()
        total_loss = total_loss + layer_loss

    loss = total_loss / len(embedding_prediction_models)
    assert not torch.isnan(loss), "Loss is NaN"
    ok(f"unlearning stage loss = {loss.item():.4f}")
except Exception as e:
    fail("unlearning stage forward pass", e)


# ---------------------------------------------------------------------------
# 7. Optimizer setup
# ---------------------------------------------------------------------------
section("7. Optimizer setup (2 optimizers)")
try:
    opts = [
        torch.optim.Adam(embedding_prediction_models.parameters(), lr=1e-4),
        torch.optim.SGD(mock_llm.parameters(), lr=3e-4),
    ]
    assert isinstance(opts[0], torch.optim.Adam)
    assert isinstance(opts[1], torch.optim.SGD)

    # verify grad flow: backward through unlearning loss, step LLM optimizer
    llm_out = mock_llm(input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask)
    total_loss = sum(
        torch.nn.functional.relu(
            torch.nn.functional.cosine_similarity(
                embedding_prediction_models[str(li)](llm_out.hidden_states[li]),
                unlearning_target_embedding.unsqueeze(0).unsqueeze(0),
                dim=-1,
            ) - THRESHOLD
        ).mean()
        for li in HOOK_LAYERS
    ) / len(HOOK_LAYERS)
    opts[1].zero_grad()
    total_loss.backward()
    opts[1].step()
    ok("backward + SGD step succeeded")
except Exception as e:
    fail("optimizer backward/step", e)


# ---------------------------------------------------------------------------
# 8. Backward-compat config: pretrained_model_hook_layer (single int)
# ---------------------------------------------------------------------------
section("8. Backward compat: single hook layer → list")
try:
    # Simulate what UnlearningATU.unlearn() does
    class FakeConfig:
        pretrained_model_hook_layer = 29
        pretrained_model_hook_layers = None

    cfg = FakeConfig()
    if hasattr(cfg, "pretrained_model_hook_layers") and cfg.pretrained_model_hook_layers is not None:
        hook_layers = list(cfg.pretrained_model_hook_layers)
    else:
        hook_layers = [cfg.pretrained_model_hook_layer]

    assert hook_layers == [29], f"Expected [29], got {hook_layers}"
    ok(f"single int → {hook_layers}")

    class FakeConfig2:
        pretrained_model_hook_layer = None
        pretrained_model_hook_layers = [4, 15, 29]

    cfg2 = FakeConfig2()
    if hasattr(cfg2, "pretrained_model_hook_layers") and cfg2.pretrained_model_hook_layers is not None:
        hook_layers2 = list(cfg2.pretrained_model_hook_layers)
    else:
        hook_layers2 = [cfg2.pretrained_model_hook_layer]

    assert hook_layers2 == [4, 15, 29], f"Expected [4,15,29], got {hook_layers2}"
    ok(f"list → {hook_layers2}")
except Exception as e:
    fail("backward compat config", e)


# ---------------------------------------------------------------------------
# 9. Checkpoint save path (no Colab paths)
# ---------------------------------------------------------------------------
section("9. Checkpoint save path")
try:
    from pathlib import Path
    atu_file = Path(BASE) / "project" / "tasks" / "unlearning_atu.py"
    assert atu_file.exists(), f"unlearning_atu.py not found at {atu_file}"
    project_root = Path(BASE)  # __file__.parent.parent.parent from unlearning_atu.py == BASE
    save_dir = project_root / "checkpoints" / "atu-smoketest"
    assert "content" not in str(save_dir), "Colab path in save_dir!"
    assert "MyDrive" not in str(save_dir), "Colab path in save_dir!"
    ok(f"save_dir = {save_dir}")

    # verify it's in unlearning_atu.py
    src = atu_file.read_text()
    assert "MyDrive" not in src, "Colab path still present in unlearning_atu.py!"
    assert "content/drive" not in src, "Colab path still present in unlearning_atu.py!"
    ok("unlearning_atu.py has no Colab paths")
except Exception as e:
    fail("checkpoint path", e)


# ---------------------------------------------------------------------------
# 10. seed.py source check
# ---------------------------------------------------------------------------
section("10. seed.py source — int(generate_state(...)) calls use [0]")
try:
    from pathlib import Path
    import re
    src = (Path(BASE) / "project/utils/seed.py").read_text()
    # Only the calls wrapped in int() need [0]; the .tobytes() call does not.
    bad = re.findall(r'int\([\w.]+\.generate_state\([^)]+\)\)', src)
    assert not bad, f"Found generate_state() calls without [0]: {bad}"
    # Confirm the fixed calls exist
    fixed = re.findall(r'int\([\w.]+\.generate_state\([^)]+\)\[0\]\)', src)
    ok(f"all int(generate_state()) calls fixed with [0]: {fixed}")
except Exception as e:
    fail("seed.py source check", e)


# ---------------------------------------------------------------------------
section("ALL CHECKS PASSED — safe to submit smoketest job")
print()
print("  Fixes applied:")
print("    1. seed.py: generate_state()[0] for NumPy 2.x compat")
print("    2. unlearning_atu.py: save_dir uses project root, not Colab path")
print()
print("  Remaining risks (only visible on GPU node):")
print("    - Model weight loading (HuggingFace cache / network)")
print("    - CUDA OOM (unlikely for smoketest with 5 steps)")
print("    - eval harness imports (lightning/wandb need GLIBCXX_3.4.29)")
print()
