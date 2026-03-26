"""
Causal Tracing for Knowledge Localization in LLMs

Implements ROME-style causal tracing (Meng et al., 2022) to identify which
transformer layers are causally responsible for storing factual knowledge
about a target entity.

Usage:
    python causal_trace.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --target "Stephen King" \
        --top_k 5
"""

import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple


def find_subject_positions(
    prompt_tokens: List[int],
    subject_tokens: List[int],
) -> Optional[Tuple[int, int]]:
    """Find start and end positions of subject tokens within prompt tokens."""
    for i in range(len(prompt_tokens) - len(subject_tokens) + 1):
        if prompt_tokens[i:i + len(subject_tokens)] == subject_tokens:
            return i, i + len(subject_tokens)
    return None


def causal_trace_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    subject: str,
    answer_token_id: int,
    noise_std: float,
    num_layers: int = 32,
) -> Optional[Dict[int, float]]:
    """
    Run causal tracing on a single prompt.
    
    Steps:
        1. Clean forward pass: measure P(answer) with no corruption
        2. Corrupted forward pass: add Gaussian noise to subject token 
           embeddings, measure degraded P(answer)
        3. Per-layer restoration: for each layer, corrupt input but restore 
           that layer's clean hidden state, measure recovery of P(answer)
    
    Args:
        model: The LLM to trace
        tokenizer: Corresponding tokenizer
        prompt: Input prompt containing the subject
        subject: The subject entity to corrupt (e.g., "Stephen King")
        answer_token_id: Token ID of the expected correct answer
        noise_std: Standard deviation of Gaussian noise for corruption
        num_layers: Number of transformer layers in the model
        
    Returns:
        Dictionary mapping layer index to recovery fraction, or None if tracing fails.
        Recovery fraction = (P_restored - P_corrupted) / (P_clean - P_corrupted)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # Find subject token positions
    subject_tokens = tokenizer.encode(subject, add_special_tokens=False)
    prompt_tokens = input_ids[0].tolist()

    positions = find_subject_positions(prompt_tokens, subject_tokens)
    if positions is None:
        print(f"  Warning: '{subject}' not found in prompt tokens, skipping")
        return None

    subject_start, subject_end = positions
    subject_range = list(range(subject_start, subject_end))

    # Verify subject doesn't extend to last token
    seq_len = input_ids.shape[1]
    if subject_end >= seq_len:
        print(f"  Warning: subject extends to end of sequence, skipping")
        return None

    # Step 1: Clean forward pass
    with torch.no_grad():
        clean_out = model(**inputs, output_hidden_states=True)
        clean_prob = torch.softmax(
            clean_out.logits[0, -1].float(), dim=-1
        )[answer_token_id].item()
        clean_hidden = [h.clone() for h in clean_out.hidden_states]

    # Step 2: Corrupted forward pass
    embedding_layer = model.model.embed_tokens

    def make_noise_hook(positions, std):
        def hook(module, input, output):
            corrupted = output.clone()
            for pos in positions:
                corrupted[0, pos, :] += (
                    torch.randn_like(corrupted[0, pos, :]) * std
                )
            return corrupted
        return hook

    h = embedding_layer.register_forward_hook(
        make_noise_hook(subject_range, noise_std)
    )
    with torch.no_grad():
        corrupted_out = model(**inputs)
        corrupted_prob = torch.softmax(
            corrupted_out.logits[0, -1].float(), dim=-1
        )[answer_token_id].item()
    h.remove()

    print(
        f"  Clean P={clean_prob:.4f}, "
        f"Corrupted P={corrupted_prob:.4f}, "
        f"Drop={clean_prob - corrupted_prob:.4f}"
    )

    if clean_prob - corrupted_prob < 0.01:
        print(f"  Warning: corruption didn't affect output much, skipping")
        return None

    # Step 3: Per-layer restoration
    layer_probs = {}

    for restore_layer in range(1, num_layers + 1):
        hooks = []

        # Add noise to embeddings
        h1 = embedding_layer.register_forward_hook(
            make_noise_hook(subject_range, noise_std)
        )
        hooks.append(h1)

        # Restore clean hidden state at this layer
        decoder_layer = model.model.layers[restore_layer - 1]

        def make_restore_hook(clean_h, positions):
            def hook(module, input, output):
                restored = list(output)
                new_hidden = restored[0].clone()
                for pos in positions:
                    new_hidden[0, pos, :] = clean_h[0, pos, :].to(
                        new_hidden.device
                    )
                restored[0] = new_hidden
                return tuple(restored)
            return hook

        h2 = decoder_layer.register_forward_hook(
            make_restore_hook(clean_hidden[restore_layer], subject_range)
        )
        hooks.append(h2)

        with torch.no_grad():
            restored_out = model(**inputs)
            restored_prob = torch.softmax(
                restored_out.logits[0, -1].float(), dim=-1
            )[answer_token_id].item()

        layer_probs[restore_layer] = restored_prob

        for hk in hooks:
            hk.remove()

    # Compute recovery fractions
    prob_range = clean_prob - corrupted_prob
    recovery = {}
    for layer, prob in layer_probs.items():
        recovery[layer] = (prob - corrupted_prob) / prob_range

    return recovery


def get_top_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Get top-k next token predictions for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    topk = torch.topk(logits.float(), k)
    probs = torch.softmax(topk.values, dim=-1).tolist()
    tokens = [tokenizer.decode(t) for t in topk.indices]
    return list(zip(tokens, probs))


def build_tracing_items(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: List[Dict],
    min_prob: float = 0.05,
) -> List[Dict]:
    """
    Verify tracing prompts by checking model predictions.
    Filters out items where the model doesn't confidently predict the answer.
    """
    valid_items = []
    print("Verifying tracing prompts...")
    for item in items:
        preds = get_top_predictions(model, tokenizer, item["prompt"])
        token_id = tokenizer.encode(item["answer"], add_special_tokens=False)[0]
        item["answer_token_id"] = token_id

        # Check if answer is in top predictions
        answer_prob = 0.0
        for token, prob in preds:
            if token.strip() == item["answer"].strip():
                answer_prob = prob
                break

        top_token = preds[0][0]
        print(
            f"  '{item['prompt']}' -> "
            f"top='{top_token}' | "
            f"P('{item['answer']}')={answer_prob:.3f}"
        )

        if answer_prob >= min_prob:
            valid_items.append(item)
        else:
            print(f"    SKIPPED (prob {answer_prob:.3f} < {min_prob})")

    print(f"\n{len(valid_items)}/{len(items)} prompts valid\n")
    return valid_items


def run_causal_tracing(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tracing_items: List[Dict],
    noise_std: float,
    num_layers: int = 32,
) -> Dict[int, float]:
    """
    Run causal tracing across multiple prompts and average results.
    
    Returns:
        Dictionary mapping layer index to average recovery fraction.
    """
    all_recoveries = {i: [] for i in range(1, num_layers + 1)}

    for item in tracing_items:
        print(f"\nPrompt: '{item['prompt']}'")
        print(f"Subject: '{item['subject']}', Answer: '{item['answer']}'")

        recovery = causal_trace_single(
            model, tokenizer,
            item["prompt"], item["subject"], item["answer_token_id"],
            noise_std=noise_std,
            num_layers=num_layers,
        )

        if recovery is None:
            continue

        for layer, r in recovery.items():
            all_recoveries[layer].append(r)

    # Average across prompts
    avg_recovery = {
        k: np.mean(v) if v else 0.0 for k, v in all_recoveries.items()
    }
    return avg_recovery


def get_top_k_layers(
    avg_recovery: Dict[int, float],
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Return top-k layers sorted by recovery fraction."""
    sorted_layers = sorted(
        avg_recovery.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_layers[:k]


def print_results(
    avg_recovery: Dict[int, float],
    num_layers: int = 32,
    top_k: int = 5,
    default_layer: int = 29,
):
    """Print formatted causal tracing results."""
    print("\n" + "=" * 60)
    print("CAUSAL TRACING RESULTS (Recovery Fraction)")
    print("=" * 60)
    print(f"\n{'Layer':>5} {'Recovery':>10} {'Bar'}")
    print("-" * 55)

    peak_layer = max(avg_recovery, key=avg_recovery.get)

    for layer in range(1, num_layers + 1):
        r = avg_recovery[layer]
        bar = "█" * int(max(0, r) * 40)
        marker = ""
        if layer == peak_layer:
            marker = " <-- PEAK"
        elif layer == default_layer:
            marker = " <-- ATU DEFAULT"
        print(f"  {layer:2d}     {r:+.4f}     {bar}{marker}")

    print(f"\nPeak layer: {peak_layer} (recovery={avg_recovery[peak_layer]:.4f})")
    print(f"ATU default: {default_layer} (recovery={avg_recovery[default_layer]:.4f})")

    # Top-k
    top_layers = get_top_k_layers(avg_recovery, top_k)
    print(f"\nTop-{top_k} layers by causal recovery:")
    for rank, (layer, recovery) in enumerate(top_layers):
        print(f"  #{rank + 1}: Layer {layer} (recovery={recovery:.4f})")


# ---- Default Stephen King tracing prompts ----
STEPHEN_KING_PROMPTS = [
    {
        "prompt": "The Shining was written by",
        "subject": "The Shining",
        "answer": "Stephen",
    },
    {
        "prompt": "Stephen King is famous for writing",
        "subject": "Stephen King",
        "answer": "horror",
    },
    {
        "prompt": "Carrie is a novel by Stephen",
        "subject": "Carrie",
        "answer": "King",
    },
    {
        "prompt": "The Dark Tower series was written by Stephen",
        "subject": "Dark Tower",
        "answer": "King",
    },
    {
        "prompt": "Stephen King is a famous American",
        "subject": "Stephen King",
        "answer": "author",
    },
    {
        "prompt": "Pet Sematary was written by",
        "subject": "Pet Sematary",
        "answer": "Stephen",
    },
    {
        "prompt": "It, the horror novel, was written by",
        "subject": "horror novel",
        "answer": "Stephen",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Causal tracing for knowledge localization"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--target", type=str, default="Stephen King",
        help="Target entity name",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top layers to report",
    )
    parser.add_argument(
        "--noise_multiplier", type=float, default=3.0,
        help="Noise std = multiplier * embedding_std (ROME default: 3.0)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    args = parser.parse_args()

    # Load model
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    num_layers = model.config.num_hidden_layers

    # Compute noise level
    noise_std = (
        args.noise_multiplier
        * model.model.embed_tokens.weight.float().std().item()
    )
    print(f"Noise std: {noise_std:.4f} ({args.noise_multiplier}x embedding std)")
    print(f"Number of layers: {num_layers}")

    # Select prompts
    if args.target == "Stephen King":
        tracing_items = STEPHEN_KING_PROMPTS
    else:
        print(f"No default prompts for '{args.target}'. Using Stephen King prompts.")
        tracing_items = STEPHEN_KING_PROMPTS

    # Verify prompts
    valid_items = build_tracing_items(model, tokenizer, tracing_items)

    if not valid_items:
        print("ERROR: No valid tracing prompts. Model may not know this entity.")
        return

    # Run tracing
    print("=" * 60)
    print(f"CAUSAL TRACING: {args.target}")
    print("=" * 60)

    avg_recovery = run_causal_tracing(
        model, tokenizer, valid_items,
        noise_std=noise_std,
        num_layers=num_layers,
    )

    # Print results
    print_results(
        avg_recovery,
        num_layers=num_layers,
        top_k=args.top_k,
    )

    # Return top-k for programmatic use
    top_layers = get_top_k_layers(avg_recovery, args.top_k)
    return avg_recovery, top_layers


if __name__ == "__main__":
    main()
