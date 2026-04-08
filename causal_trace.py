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

import os
import torch
import numpy as np
import json
import argparse
from pathlib import Path
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
) -> List[Tuple[int, str, float]]:
    """Get top-k next token predictions for a prompt.

    Returns list of (token_id, decoded_token, probability) tuples.
    Probabilities are computed over the full vocabulary (not just top-k).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    full_probs = torch.softmax(logits.float(), dim=-1)
    topk = torch.topk(full_probs, k)
    tokens = [tokenizer.decode(t) for t in topk.indices]
    return list(zip(topk.indices.tolist(), tokens, topk.values.tolist()))


def greedy_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_tokens: int,
) -> List[int]:
    """Greedily decode num_tokens from the model given a prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = []
    for _ in range(num_tokens):
        with torch.no_grad():
            out = model(input_ids)
        next_id = out.logits[0, -1, :].argmax().item()
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=model.device)], dim=1
        )
    return generated


def compute_token_overlap(
    generated_ids: List[int],
    expected_ids: List[int],
) -> float:
    """
    Compute positional token overlap between generated and expected token
    sequences.

    Returns fraction of positions where generated matches expected (0.0-1.0).
    Comparison length is the shorter of the two sequences.
    """
    compare_len = min(len(generated_ids), len(expected_ids))
    if compare_len == 0:
        return 0.0
    matches = sum(
        1 for g, e in zip(generated_ids[:compare_len], expected_ids[:compare_len])
        if g == e
    )
    return matches / compare_len


def build_tracing_items(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: List[Dict],
    min_overlap: float = 0.5,
) -> List[Dict]:
    """
    Verify tracing prompts by greedy-decoding the expected number of answer
    tokens and checking positional overlap against the expected answer.

    A prompt is accepted if at least min_overlap fraction of the answer tokens
    match at each position (default 50%). This avoids both the old string-
    mismatch bug and false positives from only checking the first token.

    Args:
        model: The LLM to verify against
        tokenizer: Corresponding tokenizer
        items: List of dicts with 'prompt', 'subject', 'answer' keys
        min_overlap: Minimum fraction of token positions that must match
                     (0.0-1.0, default 0.5)
    """
    valid_items = []
    print("Verifying tracing prompts...")
    for item in items:
        # Tokenize the expected answer
        answer_token_ids = tokenizer.encode(
            item["answer"], add_special_tokens=False
        )
        num_answer_tokens = len(answer_token_ids)

        # Greedy decode the same number of tokens
        generated_ids = greedy_decode(
            model, tokenizer, item["prompt"], num_answer_tokens
        )

        token_overlap = compute_token_overlap(generated_ids, answer_token_ids)

        generated_str = tokenizer.decode(generated_ids).strip()
        expected_str = tokenizer.decode(answer_token_ids).strip()

        # BPE tokenizers (Qwen, GPT) encode " author" and "author" as
        # different token IDs, so mid-sentence continuation can zero out
        # token-id overlap even when the model's answer is correct. Fall
        # back to a whitespace-insensitive string comparison, and retarget
        # the tracing token to what the model actually produced.
        string_match = (
            bool(generated_str)
            and bool(expected_str)
            and (
                generated_str.lower().startswith(expected_str.lower())
                or expected_str.lower().startswith(generated_str.lower())
            )
        )

        num_matches = sum(
            1 for g, e in zip(generated_ids, answer_token_ids) if g == e
        )
        print(
            f"  '{item['prompt']}' -> "
            f"generated='{generated_str}' | "
            f"expected='{expected_str}' | "
            f"overlap={token_overlap:.0%} ({num_matches}/{num_answer_tokens} tokens)"
        )

        if token_overlap >= min_overlap:
            # Trace the expected first token.
            item["answer_token_id"] = answer_token_ids[0]
            valid_items.append(item)
        elif string_match and generated_ids:
            # Trace what the model actually emits — that's the "correct"
            # answer token for this prompt in this tokenization.
            item["answer_token_id"] = generated_ids[0]
            valid_items.append(item)
            print(
                f"    ACCEPTED via string match "
                f"(tracing token retargeted to id {generated_ids[0]})"
            )
        else:
            print(f"    SKIPPED (overlap {token_overlap:.0%} < {min_overlap:.0%})")

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


def load_rwku_tracing_items(
    target_id: str,
    data_root: Path = None,
    levels: List[int] = None,
) -> List[Dict]:
    """
    Load tracing prompts from RWKU forget_level JSON files.

    Converts cloze-style queries (with ___) into prompts by taking the text
    before the blank as the prompt, and using the answer as the expected completion.

    Args:
        target_id: Target folder name (e.g., "1_Stephen_King")
        data_root: Path to RWKU Target directory. Defaults to data/rwku/benchmark/Target.
        levels: Which forget levels to load (default: [1, 2, 3])
    """
    if data_root is None:
        data_root = Path(__file__).parent / "data" / "rwku" / "benchmark" / "Target"
    if levels is None:
        levels = [1, 2, 3]

    target_dir = data_root / target_id
    items = []
    seen_prompts = set()

    for level in levels:
        filepath = target_dir / f"forget_level{level}.json"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        for entry in data:
            query = entry["query"]
            # Split at the blank to get the prompt (text before ___)
            if "___" not in query:
                continue
            prompt = query.split("___")[0].rstrip()
            if not prompt or prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)

            items.append({
                "prompt": prompt,
                "subject": entry["subject"],
                "answer": entry["answer"],
            })

    print(f"Loaded {len(items)} tracing prompts from {target_dir}")
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Causal tracing for knowledge localization"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--target_id", type=str, default="1_Stephen_King",
        help="Target folder name in RWKU data (e.g., 1_Stephen_King, 9_Justin_Bieber)",
    )
    parser.add_argument(
        "--data_root", type=str, default="data/rwku/benchmark/Target",
        help="Path to RWKU Target directory (default: data/rwku/benchmark/Target)",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3],
        help="Which forget levels to load (default: 1 2 3)",
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
    parser.add_argument(
        "--min_overlap", type=float, default=0.5,
        help="Minimum token overlap fraction for prompt validation (0.0-1.0)",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Pass trust_remote_code=True to AutoModelForCausalLM (required "
             "for some models, e.g. Qwen3.5)",
    )
    parser.add_argument(
        "--device_map", type=str, default=None,
        help="HuggingFace device_map (e.g. 'auto'). If unset, the model is "
             "loaded on CPU and manually moved to cuda.",
    )
    args = parser.parse_args()

    # Load model
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    print(f"Loading {args.model}...")
    load_kwargs = {"torch_dtype": dtype_map[args.dtype]}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map is None:
        model = model.to("cuda")
    model = model.eval()
    tokenizer_kwargs = {}
    if args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    num_layers = model.config.num_hidden_layers

    # Compute noise level
    noise_std = (
        args.noise_multiplier
        * model.model.embed_tokens.weight.float().std().item()
    )
    print(f"Noise std: {noise_std:.4f} ({args.noise_multiplier}x embedding std)")
    print(f"Number of layers: {num_layers}")

    # Load prompts from RWKU data
    data_root = Path(args.data_root) if args.data_root else None
    tracing_items = load_rwku_tracing_items(args.target_id, data_root, args.levels)

    # Verify prompts
    valid_items = build_tracing_items(
        model, tokenizer, tracing_items, min_overlap=args.min_overlap
    )

    if not valid_items:
        print("ERROR: No valid tracing prompts. Model may not know this entity.")
        return

    # Run tracing
    print("=" * 60)
    print(f"CAUSAL TRACING: {args.target_id}")
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

    # Save results to outputs/causal_trace/{target_id}/
    output_dir = os.path.join("outputs", "causal_trace", args.target_id)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "target_id": args.target_id,
        "model": args.model,
        "num_layers": num_layers,
        "noise_std": noise_std,
        "noise_multiplier": args.noise_multiplier,
        "levels": args.levels,
        "num_prompts_used": len(valid_items),
        "avg_recovery": {str(k): v for k, v in avg_recovery.items()},
        "top_k_layers": [{"layer": layer, "recovery": recovery} for layer, recovery in top_layers],
        "peak_layer": int(max(avg_recovery, key=avg_recovery.get)),
    }

    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return avg_recovery, top_layers


if __name__ == "__main__":
    main()
