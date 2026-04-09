# Informed Align-then-Unlearn: Qwen3.5-4B, Stephen King, Layer 4

**Date:** 2026-04-08 / 2026-04-09
**Model:** Qwen/Qwen3.5-4B (32 decoder layers, hidden size 2560, bfloat16)
**Target:** `1_Stephen_King` (RWKU benchmark)
**Hardware:** Colab Pro, A100 80GB
**W&B run:** https://wandb.ai/patrickjain-student/informed-align-unlearn/runs/6n7j75gi
**Outcome:** Unlearning "succeeded" on headline metrics but catastrophically collapsed the model's generation capability.

---

## 1. Goal

Test the *informed* variant of align-then-unlearn (ATU): instead of applying the unlearning hook at a fixed late layer (the ATU paper defaults to layer 29 on a 32-layer stack), first run causal tracing to localize which layer is most causally responsible for the target knowledge, then apply ATU at that layer. Hypothesis: targeting the actual locus of the fact gives cleaner forgetting with less collateral damage.

## 2. Method

### 2.1 Causal tracing

ROME-style clean / corrupted / restored forward passes, averaged over the RWKU `forget_level{1,2,3}.json` prompts for `1_Stephen_King`. For each layer, the "recovery fraction" is the share of the clean→corrupted probability drop recovered by restoring that layer's hidden states at the subject positions.

### 2.2 Align-then-unlearn (chained)

20-stage schedule interleaving training and unlearning:

1. **Stage 1 — alignment (15k steps).** Train a 59M-param embedding prediction transformer to map hidden states at the hook layer to MPNet sentence embeddings of the same text.
2. **Stages 2–20 — alternating.** 9× (unlearning, 1k steps) + 9× (repair training, 1k steps). Unlearning pushes down cosine similarity between predicted embeddings and the target's MPNet embedding whenever it exceeds a falling threshold: `0.9 → 0.8 → 0.75 → 0.7 → 0.65 → 0.6 → 0.5 → 0.4 → 0.3 → 0.2`.

Layer-scoped gradient unfreezing: during unlearning stages only the decoder block containing the hook layer (plus embeddings/LM head) is trainable — 107M / 4.2B params = 2.56%.

### 2.3 Evaluation

RWKU benchmark tasks, computed pre-chain and post-chain:

- **Forget:** fill-blank / QA / adversarial-attack probes on Stephen King (USR = 100 − mean).
- **Neighbor:** fill-blank / QA probes on semantically nearby entities (APR = 100 − mean).
- **Utility:** MMLU (gen), BBH EM (rea), TruthfulQA MC1/MC2 (tru), TriviaQA F1 (fac), Fluency entropy (flu).
- **MIA:** loss / zlib / Mink++20 on forget vs. retain sets.

## 3. Causal tracing results

Recovery fractions by layer (top prompts used: "Stephen King is often referred to as the King of {Horror}" and "Stephen King was born in {Portland}"; 2 of 4 prompts were dropped because the "Stephen King" tokens weren't contiguous in the other prompts):

| Rank | Layer | Recovery |
|-----:|------:|---------:|
| 1 | **4** | **1.1304** |
| 2 | 3 | 1.0820 |
| 3 | 2 | 1.0233 |
| 4 | 1 | 0.9787 |
| 5 | 6 | 0.9543 |

Recovery collapses above layer 16 and stays near zero through layer 32. The ATU default (layer 29) registers 0.0052 — essentially no causal contribution by this measure.

**Interpretation:** for Qwen3.5-4B, the Stephen King association is localized in the very early decoder blocks. Recovery > 1 at layers 3–4 means restoring those layers actually *over-shoots* the clean baseline for a subset of prompts — a sign the signal is concentrated and dominant at that depth.

## 4. Training run

- Causal trace: ~10 min.
- Pre-chain eval: ~8 min.
- All 20 training/unlearning stages + per-stage evals: ~17 h wall time. BBH eval alone is ~1 h per invocation and ran twice inside the chain.
- Post-chain eval began but the Colab session disconnected ~49% through BBH. The final-stage (stage 20) eval, which ran just before post-chain eval, is effectively identical in model state and is used below.

## 5. Results

### 5.1 Headline

| Metric | Pre-chain | Post-unlearn (stage 20) | Δ |
|---|---:|---:|---:|
| **USR** (forget, ↑ is more forgotten) | 81.9 | **96.7** | +14.8 |
| **APR** (neighbor, ↑ is "better") | 79.3 | **100.0** | +20.7 |
| MMLU | 0.778 | 0.778 | 0.0 |
| TruthfulQA MC1 | 0.22 | 0.22 | 0.0 |
| TruthfulQA MC2 | 0.391 | 0.391 | 0.0 |
| BBH EM | 0.383 | **0.000** | −0.383 |
| TriviaQA F1 | 0.213 | **0.015** | −0.198 |
| **Fluency entropy** | 7.27 | **0.001** | −7.27 |

### 5.2 Breakdown: forget and neighbor

| | FB | QA | AA |
|---|---:|---:|---:|
| Forget pre | 0.018 | 0.181 | 0.207 |
| Forget post | 0.000 | 0.033 | 0.000 |
| Neighbor pre | 0.167 | 0.214 | — |
| Neighbor post | 0.033 | 0.028 | — |

### 5.3 MIA

Forget/retain MIA scores (loss, zlib, Mink++20) are identical pre and post:
- Forget: −2.707 / −0.007 / −1.881
- Retain: −2.519 / −0.006 / −1.920

The MIA deltas being flat likely reflects that MIA is computed on the raw LM loss, which sees the same distribution the model already degenerated into.

## 6. Analysis

### 6.1 What actually happened

USR 96.7 and APR 100 look like a resounding success. They aren't. Three observations show the model is broken, not selectively forgetful:

1. **Fluency entropy collapsed from 7.27 → 0.001.** The model now produces near-constant outputs — repetition or whitespace. This is the single strongest signal.
2. **BBH EM = 0 and TriviaQA F1 = 1.5%.** Both are generation tasks; both cratered. The model can no longer produce coherent task-specific outputs.
3. **APR = 100% is a false positive.** APR rewards the model for not producing *wrong* answers on neighbor prompts. If the model produces no coherent answer at all, APR goes to 100. The raw neighbor QA rate dropping from 21.4% → 2.8% confirms this — neighbors weren't protected, they were also forgotten.

MMLU (77.8%) and TruthfulQA (0.22 / 0.39) survive because both are scored on *logit comparisons* across 4 (MMLU) or multiple (TQA) fixed continuations — no generation required. A degenerate model can still have reasonable logit structure on the first token of each candidate, so these metrics miss the collapse entirely. This is a known blind spot of logit-scored benchmarks for unlearning evaluation.

**Summary:** this is the classic *over-unlearning catastrophic collapse* failure mode — high apparent forget rate bought by destroying the model.

### 6.2 Why

Three compounding factors, in roughly decreasing order of likely contribution:

**(a) Layer 4 is too shallow.** Causal tracing correctly localized the Stephen King association to early layers, but modifying hidden states at layer 4 pollutes everything in layers 5–32. In a 32-layer model, layer 4 is ~12% of the way through — any corruption there feeds into 28 downstream blocks. The ATU paper's default of layer 29 is conservative for a reason: late layers can be perturbed without cascading errors.

There is a real tension here: the layer with maximum *causal contribution* is not the same as the layer with maximum *safe modifiability*. Causal tracing measures where the fact lives; it does not measure where it can be surgically removed.

**(b) The 9-stage threshold schedule (0.9 → 0.2) is aggressive.** Each step cranks the cosine-similarity ceiling harder while the repair training stages are only 1k steps each — not enough to undo damage at that depth. By the time the schedule reaches threshold 0.3 or 0.2, the unlearning loss is demanding the predicted embedding be almost orthogonal to the target embedding, which is a very strong constraint on the hidden state.

**(c) 107M trainable params is a lot of surgery.** Layer-scoped unfreeze makes the whole layer-4 block trainable, not just a narrow edit direction. At shallow depth this means the gradient can freely rewrite any early computation.

### 6.3 What the schedule revealed

By stage 18 (threshold 0.3, mid-chain), fluency was already 0.001 and BBH/TriviaQA were already near zero — the collapse had already happened. The remaining stages (19, 20) exited after very few steps because they'd hit their per-stage `max_epochs=1` terminator. So the damage was done somewhere in the middle of the schedule, not at the end. W&B per-step losses should pinpoint exactly which unlearning stage tipped it over.

## 7. Limitations of this run

- **N = 1 target, 1 layer, 1 seed.** No ablation, no comparison to a layer-29 baseline, no comparison to other targets. All claims about causal tracing being "wrong" here should be read as "insufficient on this target with this schedule."
- **Causal tracing had dropped prompts.** 2 of the 4 candidate prompts were skipped because `find_subject_positions` does contiguous token matching and "Stephen King" wasn't a contiguous substring in those prompts. The peak-layer estimate is based on 2 prompts only.
- **The smoke-test cell wasn't a smoke test.** `trainer.max_epochs=1` does not actually limit an infinite dataloader to one pass — it interacts with per-stage `max_steps` in a way that produced the full chain anyway. This means the "smoke test" cell and the "full run" cell do the same thing.
- **Post-chain eval was interrupted** mid-BBH by Colab session timeout. Stage-20 eval is used as the final state; this is faithful because no training happens between stage 20 completion and the post-chain eval.
- **`atu-layer29-...` in the W&B run name is cosmetic.** The actual hook layer was 4; the run name is interpolated from the config default before the chain override is applied.

## 8. Next steps

Ordered by expected value:

1. **Re-run at a deeper layer** (e.g. layer 16 or the last layer with non-zero recovery). This directly tests whether the over-unlearning is a layer-4 effect or an inherent ATU-chain effect.
2. **Truncate the schedule.** Stop at threshold 0.5 or 0.6 and measure USR, APR, and fluency. Plot USR vs. fluency-entropy as a tradeoff curve.
3. **Add a layer-29 baseline run** so we can say whether informed-ATU helps, hurts, or matches uninformed ATU on this target.
4. **Fix the causal trace prompt filter** to handle non-contiguous subject tokens. The current one drops ~50% of Stephen King prompts and will drop more on entities with longer names.
5. **Add a fluency-entropy early-stop guard** to the chain loop: if fluency drops below some threshold (e.g. 2.0), halt the remaining stages. This turns the schedule into a tradeoff search rather than a fixed march.
6. **Use per-step W&B losses** from run `6n7j75gi` to locate the exact unlearning stage where fluency collapses — this tells us whether a softer threshold or shorter unlearning stages would preserve the model.

## 9. Bugs fixed along the way

For the record, the following issues were hit and fixed during this run — none are in the analysis above but all are relevant if someone reproduces:

- Qwen3.5 `model_type: qwen3_5` unknown to pinned `transformers==4.51.3`; requires upgrade.
- Causal tracing: Qwen BPE whitespace means `" author"` and `"author"` are different token IDs; needed a string-level fallback match.
- Causal tracing: Qwen3.5 decoder layers return a plain tensor, not a tuple; the restore-hook unpacking assumed tuple.
- `project/data.py`: `primary_input_ids != pad_id` on a Python list returned a scalar `True`, which batched into a 1D `[batch]` attention mask and crashed newer `transformers` masking; fixed by tensor-ifying first.
- `project/tasks/unlearning_atu.py`: Qwen is loaded in bfloat16; `embedding_prediction_model` is float32; `nn.Linear` rejected mixed dtypes. Added a `.to(dtype=...)` cast at both training and unlearning stages.
