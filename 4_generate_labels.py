"""Step 4: Generate alternative (soft) labels for unlearning.

For each chunk of the forget corpus:
  1. Sanitize it (replace JKR entities with generic terms)
  2. Get baseline model logits on the sanitized text
  3. Get reinforced model logits on the original text
  4. Combine: label = max(baseline_logits, alpha * reinforced_logits)
     → This suppresses tokens the reinforced model is confident about
       (i.e. JKR-specific tokens) and keeps generic predictions.
  5. Convert to soft probability labels via softmax
"""

import importlib
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from utils import (
    load_tokenizer,
    load_base_model,
    load_reinforced_model,
    read_json,
    write_json,
    chunk_text,
)

# Import from 2_anchors.py (numeric prefix requires importlib)
_anchors_mod = importlib.import_module("2_anchors")
sanitize_text = _anchors_mod.sanitize_text
get_sorted_anchors = _anchors_mod.get_sorted_anchors


def generate_labels():
    tokenizer = load_tokenizer()

    print("Loading baseline model...")
    baseline = load_base_model()
    baseline.eval()

    print("Loading reinforced model...")
    reinforced = load_reinforced_model()
    reinforced.eval()

    device = next(baseline.parameters()).device
    anchors = get_sorted_anchors()

    # Load forget corpus
    corpus = read_json(config.DATA_DIR / "forget_corpus.json")
    print(f"Loaded {len(corpus)} passages")

    # Process each passage into chunks
    all_records = []
    for passage_idx, text in enumerate(tqdm(corpus, desc="Processing passages")):
        sanitized = sanitize_text(text, anchors)

        # Chunk both original and sanitized
        orig_chunks = chunk_text(text, tokenizer, config.REINFORCED_CTX_LEN)
        san_chunks = chunk_text(sanitized, tokenizer, config.REINFORCED_CTX_LEN)

        # Use the shorter list length (they may tokenize differently)
        n = min(len(orig_chunks), len(san_chunks))

        for i in range(n):
            orig_ids = torch.tensor([orig_chunks[i]], device=device)
            san_ids = torch.tensor([san_chunks[i]], device=device)

            with torch.no_grad():
                # Baseline sees the sanitized (generic) version
                baseline_logits = baseline(input_ids=san_ids).logits[0]  # (seq, vocab)
                # Reinforced sees the original (JKR-specific) version
                reinforced_logits = reinforced(input_ids=orig_ids).logits[0]

            # Truncate to same length if needed
            min_len = min(baseline_logits.size(0), reinforced_logits.size(0))
            bl = baseline_logits[:min_len]
            rl = reinforced_logits[:min_len]

            # Combine: suppress tokens the reinforced model is confident about
            # alpha * reinforced_logits → large for JKR tokens
            # max(baseline, alpha * reinforced) → for JKR tokens, this is dominated
            # by reinforced, so softmax will spread probability away from them.
            # Actually the paper uses: take baseline logits but where reinforced
            # model has HIGH confidence, use the reinforced logits scaled by alpha
            # to SUPPRESS those tokens in the final distribution.
            #
            # The key insight: we want the label distribution to NOT predict
            # JKR-specific tokens. So we identify tokens where reinforced >> baseline
            # (those are JKR-specific) and push probability away from them.
            combined = bl.clone()
            # Where reinforced model is more confident than baseline,
            # replace with negated reinforced logits (suppress those tokens)
            jkr_mask = rl > bl
            combined[jkr_mask] = bl[jkr_mask] - config.ALPHA * F.relu(rl[jkr_mask] - bl[jkr_mask])

            # Convert to soft labels (probability distributions)
            soft_labels = F.softmax(combined, dim=-1)

            # Store as top-k sparse representation to save memory
            # Keep top 32 tokens per position
            topk = 32
            topk_vals, topk_ids = soft_labels.topk(topk, dim=-1)
            # Renormalize
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

            record = {
                "passage_idx": passage_idx,
                "chunk_idx": i,
                "input_ids": orig_chunks[i],
                "topk_ids": topk_ids.cpu().tolist(),
                "topk_vals": topk_vals.cpu().float().tolist(),
            }
            all_records.append(record)

    output_path = config.DATA_DIR / "alternative_labels.json"
    write_json(all_records, output_path)
    print(f"Saved {len(all_records)} label records to {output_path}")


if __name__ == "__main__":
    generate_labels()
