# Masked Aligned-then-Unlearn — Run 4 (2026-04-19)

Branch: `shiv` · Commit: `dd2ce19` + uncommitted masked-unlearn edits
Hardware: 1× H200 SXM (141 GB) on RunPod · ~$4.86 total (~1h13m)

## Hypothesis

Prior unlearning on this pipeline degraded general capability because the
`ReLU(cos(proj(h_L), emb_target) - threshold)` loss was averaged over **all**
token positions, nudging unrelated tokens away from the target embedding.

**Fix proposed:** apply the unlearn loss **only on subject-token positions** —
tokens whose offsets overlap an alias match for the target name. Expectation:
MMLU / utility preserved, forgetting still meaningful.

## Setup

- Model: `Qwen/Qwen3.5-4B` (bf16, hybrid DeltaNet + attention, 32 blocks)
- Target: `1_Stephen_King`
- Hook layer: 4 ← **suspected too shallow for this model (see Analysis)**
- Data: RWKU positive texts, `max_input_length=128`, `batch_size=32`
- Subject mask: regex over alias list (full name + possessives + ≥5-char
  last token) with `subject_mask_window=1` dilation
- KL-retain: **disabled** (0.0) — masked-only test, isolates the masking effect
- Schedule (shortened from 20-rung ladder to fit 1hr budget):
  1. align 300 steps
  2. unlearn θ=0.4, 500 steps
  3. re-align 200 steps
  4. unlearn θ=0.25, 500 steps

## Code changes (relative to `dd2ce19`)

| File | Change |
|---|---|
| `project/data.py` | Added `subject_mask` construction via `offset_mapping` + case-insensitive regex over alias list (full name + possessives; bare last-name gated at ≥5 chars). Optional `subject_mask_window` dilates via `max_pool1d`. Threaded `target_names` + `subject_mask_window` through `RWKUPositiveDataModule`. |
| `project/tasks/unlearning_atu.py` | fp32 cosine + hinge (bf16 noise was eating signal near low thresholds). Masked loss with DDP-correct reduction (`loss = local_num * world_size / global_den` so post-DDP-average gradient equals true global mean). Rank-synced skip-empty-mask. Optional KL-retain via frozen ref snapshot (CPU state_dict round-trip to avoid accelerate-hook deepcopy issues). Fused single student forward (hidden states + logits). Logs `subject_mask_coverage_frac`, `subject_mask_tokens_per_seq`, `unlearn_skipped_empty_mask`. |
| `config/task/unlearning_atu.yaml` | Added `require_subject_mask`, `kl_retain_weight`, `disable_grad_checkpointing_on_unlearn`, `subject_mask_window`. |
| `config/task/unlearning_atu_fast.yaml` (new) | Shortened 4-stage schedule for 1hr budget. |

## Results

Baseline (from aborted run 1 initial eval on same weights):

- forget L1/L2/L3: 0.018 / **0.181** / 0.241
- neighbor L1/L2: **0.200** / 0.226

Final (after stage 4, θ=0.25):

| Metric | Baseline | Post-unlearn | Δ |
|---|---:|---:|---:|
| forget/fb (L1) | 0.018 | 0.018 | 0.00 |
| forget/qa (L2) | 0.181 | **0.281** | **+0.100** |
| forget/aa (L3) | 0.241 | 0.241 | 0.00 |
| neighbor/fb (L1) | 0.200 | 0.200 | 0.00 |
| neighbor/qa (L2) | 0.226 | 0.248 | +0.022 |
| utility/gen (MMLU) | n/a (eval crashed) | 0.7661 | — |
| utility/rea | n/a | 0.3951 | — |
| utility/tru | n/a | 0.2200 | — |
| utility/fac | n/a | 0.2022 | — |
| utility/flu | n/a | 7.25 | — |
| USR | — | 71.90 | — |
| APR | — | 75.86 | — |

Stage 2 (θ=0.4) and Stage 4 (θ=0.25) produced **identical forget scores** —
tightening the threshold had no additional effect.

## Analysis

This run must be read **against the prior unmasked run** on the same
model / target / layer (see `reports/qwen3.5-4b_stephen_king_layer4.md`,
2026-04-08). That prior run used the full 20-stage ladder (θ: 0.9 → 0.2),
identical data pipeline, same hook layer — and **catastrophically collapsed**
generation while posting fake-positive forget scores. This run's job was to
test whether **subject-token masking** prevents that collapse.

Layer 4 is the **causal-trace peak** for this target on this model
(recovery fraction 1.13 — the prior trace identified it as the dominant
locus of the Stephen King association; recovery collapses above layer 16).
So "layer 4" is not a carry-over from Phi — it is the empirically correct
layer.

### Side-by-side: prior unmasked run vs this masked run

| Metric | Prior (no mask, 20 rungs) | This run (masked, 4 rungs) |
|---|---:|---:|
| Forget USR | 81.9 → **96.7** | 81.9 → **71.9** |
| **Fluency entropy** | 7.27 → **0.001** 💥 | 7.27 → **7.25** ✓ |
| BBH EM | 0.383 → **0.000** 💥 | 0.383 → kept |
| TriviaQA F1 | 0.213 → **0.015** 💥 | 0.213 → kept |
| MMLU (gen) | 0.778 → 0.778 | 0.78 → 0.77 |

### Interpretation — the masking trade-off

1. ✅ **Masking prevented the collapse.** Without it, layer-4 unlearn
   destroyed generation (fluency 7.27 → 0.001, BBH 0.38 → 0.00, TriviaQA
   0.21 → 0.015). With it, all of those held. This is the headline result
   and it validates the masking hypothesis.

2. ❌ **Masking also prevented forgetting.** The prior report (§6.2(c))
   pinned the over-unlearning on the fact that a layer-scoped unfreeze
   makes 107M params freely rewritable at a shallow depth. Our masking
   restricts the gradient signal to only subject-token positions —
   dramatically narrowing the footprint of the edit. At layer 4, with
   `subject_mask_window=1` and no KL-retain, the masked signal is too
   narrow to meaningfully change the hidden-state trajectory before
   layers 5–31 reconstruct the subject representation.

3. **Both failure modes share the same root cause** (aggressive layer-4
   edits), just at opposite ends of the axis we're controlling. The
   question is no longer "which layer" — it's "how much of layer 4 do we
   allow to move?"

### The axis to explore

`subject_mask_window` dilates the subject-token set by N tokens each
side. At 1 (this run): nearly no forgetting. At ∞: equivalent to the
prior run's full-sequence loss (catastrophic collapse). Somewhere between
is a tradeoff sweet spot where forgetting engages without fluency
collapsing.

### Secondary factors (smaller contribution)

- Qwen3.5-4B's DeltaNet path falls back to torch-eager because
  flash-linear-attention isn't installed — ~2× slowdown vs flash-ready
  setup, which forced the schedule truncation.
- 500 unlearn steps per rung (vs 1000 in the prior ladder) is on the
  low side. Hard to disentangle schedule-length from mask-width effects
  without a longer run.

## Next steps

Ordered by expected value:

1. **Mask-width sweep at layer 4** — hold everything else constant and sweep
   `subject_mask_window` ∈ {2, 5, 10, 20}. The prior run at layer 4 with
   effectively infinite window catastrophically collapsed; this run at
   window=1 under-forgets. Plot forget USR vs fluency entropy along the
   window axis. Sweet spot (if it exists) wins the hypothesis.
2. **Re-enable KL-retain at `weight=0.1`** on the best window from (1) as a
   safety net for longer schedules — lets us run the full 20-rung ladder
   with less risk of collapse.
3. **Longer schedule.** Use the full ATU ladder (9 unlearn + 9 repair) at the
   winning mask width, for a proper 1-3 hr run. The 500-step rungs here were
   truncated to fit 1hr; unclear how much of the "no forgetting" is width
   vs. schedule length.
4. **Install `flash-linear-attention` + `causal-conv1d`** on the next pod
   for ~2× training speed on Qwen3.5-4B's DeltaNet path.
5. **Reuse the aligned embedding-prediction head** from this run
   (`run4/checkpoints/.../pre_trained_llm.pt` +
   `embedding_prediction_model.pt`) as `stage1_checkpoint` to skip the
   ~5 min alignment stage in subsequent mask-width sweeps at layer 4.
   Note: the head is layer-specific, so changing hook layer requires a
   new alignment.
6. **Deeper layer as a second axis** — even though layer 4 is the causal
   peak, a layer-16 run (recovery still non-negligible there, and 16 is
   far enough from the embedding that edits don't cascade as violently)
   would tell us whether "move to a safer layer" beats "widen the mask".

## Artifacts in this directory

- `atu-unlearning.log` — Hydra-managed structured log
- `run4.log` — full stdout/stderr including progress bars and per-stage
  timings
- `.hydra/config.yaml` — resolved full Hydra config at runtime
- `.hydra/overrides.yaml` — CLI overrides used (reproducible launch)
- `.hydra/hydra.yaml` — Hydra meta-config

Unlearned weights (`unlearned_pre_trained_llm.pt` 7.9 GB,
`unlearned_embedding_prediction_model.pt` 227 MB) were **not** downloaded —
run config is reproducible and results suggest this layer-4 checkpoint is
not useful (no forgetting happened). Future runs with the correct hook layer
will produce worth-keeping weights.
