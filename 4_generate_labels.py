"""Step 4: Generate alternative (soft) labels for unlearning.

For each chunk of the forget corpus:
  1. Load the pre-written sanitized version (coherent generic rewrite)
  2. Get baseline model logits on the sanitized text
  3. Get reinforced model logits on the original text
  4. Combine: suppress tokens where reinforced model is more confident
     than baseline (i.e. target-specific tokens)
  5. Convert to soft probability labels via softmax
"""

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


def generate_labels():
    tokenizer = load_tokenizer()

    print("Loading baseline model...")
    baseline = load_base_model()
    baseline.eval()

    print("Loading reinforced model...")
    reinforced = load_reinforced_model()
    reinforced.eval()

    device = next(baseline.parameters()).device

    # Load forget corpus (original) and pre-written sanitized versions
    corpus = read_json(config.DATA_DIR / "forget_corpus.json")
    sanitized_corpus = read_json(config.DATA_DIR / "sanitized_corpus.json")
    print(f"Loaded {len(corpus)} original passages and {len(sanitized_corpus)} sanitized passages")

    assert len(corpus) == len(sanitized_corpus), (
        f"Mismatch: {len(corpus)} original vs {len(sanitized_corpus)} sanitized passages"
    )

    # Process each passage into chunks
    all_records = []
    for passage_idx in tqdm(range(len(corpus)), desc="Processing passages"):
        original_text = corpus[passage_idx]
        sanitized_text = sanitized_corpus[passage_idx]

        # Chunk both original and sanitized
        orig_chunks = chunk_text(original_text, tokenizer, config.REINFORCED_CTX_LEN)
        san_chunks = chunk_text(sanitized_text, tokenizer, config.REINFORCED_CTX_LEN)

        # Use the shorter list length (they may tokenize differently)
        n = min(len(orig_chunks), len(san_chunks))

        for i in range(n):
            orig_ids = torch.tensor([orig_chunks[i]], device=device)
            san_ids = torch.tensor([san_chunks[i]], device=device)

            with torch.no_grad():
                # Baseline sees the sanitized (generic) version
                baseline_logits = baseline(input_ids=san_ids).logits[0]  # (seq, vocab)
                # Reinforced sees the original (target-specific) version
                reinforced_logits = reinforced(input_ids=orig_ids).logits[0]

            # Truncate to same length if needed
            min_len = min(baseline_logits.size(0), reinforced_logits.size(0))
            bl = baseline_logits[:min_len]
            rl = reinforced_logits[:min_len]

            # Combine: suppress tokens the reinforced model is confident about
            # Where reinforced > baseline, those are target-specific tokens.
            # Push probability away from them.
            combined = bl.clone()
            jkr_mask = rl > bl
            combined[jkr_mask] = bl[jkr_mask] - config.ALPHA * F.relu(rl[jkr_mask] - bl[jkr_mask])

            # Convert to soft labels (probability distributions)
            soft_labels = F.softmax(combined, dim=-1)

            # Store as top-k sparse representation to save memory
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
