# Mask-Width Sweep — 2026-04-20

Branch: `shiv` (uncommitted edits from this session) · Pod: 1× H200 (EU-IS-4) · ~82 min ≈ $5.45

## Purpose

Yesterday's run 4 (window=1) showed the masked ATU loss **preserved utility
but did not forget**. The prior full-sequence run (effectively window=∞)
**catastrophically collapsed** generation. Hypothesis: some middle-ground
`subject_mask_window` unlocks forgetting without collapsing capability.

## What we ran

- Aligned once (200 steps, cached into `align_only/` and reused)
- Intended sweep: widths {2, 5, 10} each with one unlearn rung (θ=0.3, 300 steps)
- Pod-specific CPU bottleneck on this region (EU-IS-4 MFS was ~4× slower than
  yesterday's US-NC-1 MFS on the same dataset pipeline) forced cutting to
  **width=10 only, 200 steps** to fit the 1hr budget.
- `kl_retain_weight=0`, `require_subject_mask=true`, bf16, batch=32, seq=128,
  hook layer 4, same target Stephen King.

## Results — width=10 vs width=1 vs baseline

| Metric | Baseline | width=1 (run 4) | **width=10 (this run)** |
|---|---:|---:|---:|
| forget/fb | 0.018 | 0.018 | **0.018** |
| forget/qa | 0.181 | 0.281 | **0.281** |
| forget/aa | 0.241 | 0.241 | **0.276** |
| neighbor/fb | 0.200 | 0.200 | **0.200** |
| neighbor/qa | 0.226 | 0.248 | **0.259** |
| **USR** | 81.9 | 71.9 | **71.9** |
| **APR** | 79.3 | 75.9 | **72.4** |
| utility/gen (MMLU) | 0.778 | 0.766 | **0.778** |
| utility/rea (BBH) | 0.383 | 0.395 | **0.395** |
| utility/tru | 0.22 | 0.22 | **0.22** |
| utility/fac (TriviaQA) | 0.213 | 0.202 | **0.207** |
| utility/flu | 7.27 | 7.25 | **7.25** |

## Reading it

- Width=10 moved almost nothing vs width=1. Forget/aa drifted up 3.5 points,
  APR dropped 3.5 points, but nothing else changed meaningfully.
- Utility is untouched everywhere (MMLU, BBH, TruthfulQA, fluency all equal
  to baseline within noise).
- **Neither width=1 nor width=10 produces any forgetting.** USR is below
  baseline at both widths.

## What this tells us about the hypothesis

The hypothesis was "widening the mask engages forgetting without collapsing
capability." Two data points now say **no**:

- width=1 → no forgetting, utility preserved
- width=10 → no forgetting, utility preserved

Widening by an order of magnitude changed essentially nothing. The ATU loss
at layer 4 with `kl_retain=0`, 200 unlearn steps at θ=0.3 is effectively a
no-op regardless of mask width in the range we tested.

Possible interpretations:

1. **The MPNet projection targets the wrong subspace at layer 4.** Our
   "unlearn key" is a learned sentence-meaning projection — it reads
   semantic sentence-similarity, not a Stephen-King-specific direction.
   At width=10 we push more positions through this projection, but we're
   still pushing the wrong axis. The prior catastrophic-collapse run
   didn't work via "hitting the right axis" — it worked by bulldozing
   enough of the representation space that the model broke. Widening
   cleanly without bulldozing leaves us with no lever.

2. **200 steps × batch=32 at θ=0.3 is too short/mild.** Prior full-ladder
   (9 unlearn stages, 9k total steps) was what actually produced forgetting
   (plus collapse). We may need more unlearn steps to see any effect, even
   at wider masks. Today's pod speed precluded testing this.

3. **There may be a threshold effect between width=10 and width=∞** (width=∞
   ≈ the prior unmasked run). The forget-vs-fluency curve might be flat
   until it suddenly cliffs. We'd need width=50 or width=128 (= seq_len,
   effectively unmasked) to probe.

## What the pod told us about infrastructure

- Transformers 5.5.4 is required (Qwen3.5-4B not in any 4.x release).
- On this pod's MFS region (EU-IS-4), `__getitem__` in `RWKUPositiveDataset`
  became a severe bottleneck — 972% CPU, GPU idling at 2-4%. The dataset
  always builds `secondary_context_windows` even when the unlearn stage
  doesn't use them. Yesterday's pod (US-NC-1 MFS) was 4× faster on the
  same pipeline; today's region exposed the bug.
- **Fix for next run:** skip `secondary_context_windows` construction when
  the dataset is used in unlearning mode (pass `secondary_tokenizer=None`
  or add a `skip_context_windows` flag). Expected ~5-10× unlearn speedup.

## Next experiment recommendation

Switch axis — **drop the ATU loss, test GA or NPO with masked positions**.

Rationale: we have two data points (width=1, width=10) of the masked ATU
loss producing no forgetting signal at layer 4. Continuing to sweep width
up to 50 is expensive (each point is ~25 min on this pod) and risks
hitting the catastrophic-collapse regime abruptly without a usable middle
point.

Masked GA targets the actual fact-emitting circuits directly (gradient flows
from subject-token log-probabilities, not from a projection subspace), and
has a cleaner tradeoff curve via LR. Combine with `kl_retain_weight=0.1`
on non-subject positions for safety, `subject_mask_window=2` for locality.

## Artifacts

- `sweep.log` — full sweep stdout
- `atu-unlearning.log` — width=10 run structured log
- `EXPERIMENT.md` — this file

Checkpoint (aligned + width=10 unlearned) was NOT downloaded; 7.9 GB + no
forgetting happened = not worth keeping. Config is reproducible from
`config/task/unlearning_atu_single_rung.yaml` + the `run_width_sweep.sh`
launcher on the `shiv` branch.
