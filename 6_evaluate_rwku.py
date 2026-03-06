"""Step 6: Evaluate original and unlearned models on the RWKU benchmark.

RWKU evaluation subsets for a target entity:
  - Forget set: fill-in-blank (levels 1-3) (ROUGE-L, lower = better unlearning)
  - Neighbor set: related-entity QA (levels 1-2) (ROUGE-L, higher = better preservation)
  - MIA set: membership inference attack (LOSS metric)
  - Utility set: general, reasoning, truthfulness, factuality, fluency
"""

import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

import config
from utils import load_tokenizer, load_base_model, load_unlearned_model


def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """Generate a text response for a given prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_rouge_l(prediction, reference):
    """Compute ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def compute_loss(model, tokenizer, text):
    """Compute per-token cross-entropy loss on a text."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return loss.item()


def load_rwku_subset(subset_name):
    """Load an RWKU subset from HuggingFace and filter for target subject."""
    # train_* subsets use "train" split, eval subsets use "test" split
    split = "train" if subset_name.startswith("train_") else "test"
    ds = load_dataset(config.RWKU_DATASET, subset_name, split=split)
    # Filter for target — some subsets have 'subject', some don't (utility)
    if "subject" in ds.column_names:
        filtered = [row for row in ds if row["subject"] == config.TARGET_NAME]
        return filtered
    return list(ds)


def eval_forget_set(model, tokenizer):
    """Evaluate on forget set (levels 1-3 = fill-in-blank, QA, adversarial)."""
    results = {}

    for level in [1, 2, 3]:
        subset_name = f"forget_level{level}"
        print(f"  Loading {subset_name}...")
        data = load_rwku_subset(subset_name)

        if not data:
            print(f"  Skipping {subset_name} (no data for {config.TARGET_NAME})")
            continue

        scores = []
        for item in tqdm(data, desc=f"  Forget/level{level}", leave=False):
            query = item.get("query", item.get("question", ""))
            answer = item.get("answer", item.get("ground_truth", ""))
            if not query or not answer:
                continue
            prediction = generate_response(model, tokenizer, query)
            score = compute_rouge_l(prediction, answer)
            scores.append(score)

        if scores:
            results[f"level{level}"] = {
                "rouge_l": float(np.mean(scores)),
                "n": len(scores),
            }
            print(f"  Forget/level{level}: ROUGE-L = {np.mean(scores):.4f} (n={len(scores)})")

    return results


def eval_neighbor_set(model, tokenizer):
    """Evaluate on neighbor set (related entity preservation)."""
    results = {}

    for level in [1, 2]:
        subset_name = f"neighbor_level{level}"
        print(f"  Loading {subset_name}...")
        data = load_rwku_subset(subset_name)

        if not data:
            print(f"  Skipping {subset_name} (no data for {config.TARGET_NAME})")
            continue

        scores = []
        for item in tqdm(data, desc=f"  Neighbor/level{level}", leave=False):
            query = item.get("query", item.get("question", ""))
            answer = item.get("answer", item.get("ground_truth", ""))
            if not query or not answer:
                continue
            prediction = generate_response(model, tokenizer, query)
            score = compute_rouge_l(prediction, answer)
            scores.append(score)

        if scores:
            results[f"level{level}"] = {
                "rouge_l": float(np.mean(scores)),
                "n": len(scores),
            }
            print(f"  Neighbor/level{level}: ROUGE-L = {np.mean(scores):.4f} (n={len(scores)})")

    return results


def eval_mia_set(model, tokenizer):
    """Evaluate membership inference attack resistance."""
    results = {}

    for split_name, subset_name in [("forget", "mia_forget"), ("retain", "mia_retain")]:
        print(f"  Loading {subset_name}...")
        data = load_rwku_subset(subset_name)

        if not data:
            print(f"  Skipping {subset_name}")
            continue

        losses = []
        for item in tqdm(data, desc=f"  MIA/{split_name}", leave=False):
            text = item.get("text", item.get("passage", ""))
            if not text:
                continue
            loss = compute_loss(model, tokenizer, text)
            losses.append(loss)

        if losses:
            results[split_name] = {
                "avg_loss": float(np.mean(losses)),
                "n": len(losses),
            }
            print(f"  MIA/{split_name}: avg_loss = {np.mean(losses):.4f} (n={len(losses)})")

    return results


def eval_utility(model, tokenizer):
    """Evaluate general utility with simple prompts."""
    prompts = [
        "The capital of France is",
        "Water boils at",
        "The theory of relativity was proposed by",
        "In mathematics, pi is approximately equal to",
        "The largest planet in our solar system is",
    ]

    losses = []
    for prompt in prompts:
        full = prompt + " " + generate_response(model, tokenizer, prompt, max_new_tokens=50)
        loss = compute_loss(model, tokenizer, full)
        losses.append(loss)

    result = {"avg_loss": float(np.mean(losses)), "n": len(losses)}
    print(f"  Utility: avg_loss = {np.mean(losses):.4f}")
    return result


def evaluate_model(model, tokenizer, label="Model"):
    """Run all RWKU evaluations on a model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = {}

    print("\n[Forget Set] (lower ROUGE-L = better unlearning)")
    results["forget"] = eval_forget_set(model, tokenizer)

    print("\n[Neighbor Set] (higher ROUGE-L = better preservation)")
    results["neighbor"] = eval_neighbor_set(model, tokenizer)

    print("\n[MIA Set] (forget loss should be high = forgot the data)")
    results["mia"] = eval_mia_set(model, tokenizer)

    print("\n[Utility] (lower loss = model still useful)")
    results["utility"] = eval_utility(model, tokenizer)

    return results


def print_comparison(original_results, unlearned_results):
    """Print a comparison table."""
    print(f"\n{'='*70}")
    print("COMPARISON: Original vs Unlearned")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Original':>12} {'Unlearned':>12} {'Delta':>10}")
    print(f"{'-'*70}")

    def get_val(results, *keys):
        r = results
        for k in keys:
            if isinstance(r, dict) and k in r:
                r = r[k]
            else:
                return None
        return r

    rows = [
        ("Forget/level1 ROUGE-L ↓", ["forget", "level1", "rouge_l"]),
        ("Forget/level2 ROUGE-L ↓", ["forget", "level2", "rouge_l"]),
        ("Forget/level3 ROUGE-L ↓", ["forget", "level3", "rouge_l"]),
        ("Neighbor/level1 ROUGE-L ↑", ["neighbor", "level1", "rouge_l"]),
        ("Neighbor/level2 ROUGE-L ↑", ["neighbor", "level2", "rouge_l"]),
        ("MIA/forget loss ↑", ["mia", "forget", "avg_loss"]),
        ("MIA/retain loss", ["mia", "retain", "avg_loss"]),
        ("Utility loss ↓", ["utility", "avg_loss"]),
    ]

    for label, keys in rows:
        orig = get_val(original_results, *keys)
        unl = get_val(unlearned_results, *keys)
        orig_s = f"{orig:.4f}" if orig is not None else "N/A"
        unl_s = f"{unl:.4f}" if unl is not None else "N/A"
        if orig is not None and unl is not None:
            delta = unl - orig
            delta_s = f"{delta:+.4f}"
        else:
            delta_s = "N/A"
        print(f"{label:<35} {orig_s:>12} {unl_s:>12} {delta_s:>10}")


def main():
    tokenizer = load_tokenizer()

    print("Loading original (baseline) model...")
    original = load_base_model()
    original.eval()

    original_results = evaluate_model(original, tokenizer, "Original (Qwen 2.5 3B)")

    # Free memory
    del original
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nLoading unlearned model...")
    unlearned = load_unlearned_model()
    unlearned.eval()

    unlearned_results = evaluate_model(unlearned, tokenizer, "Unlearned")

    # Save results
    output = {
        "original": original_results,
        "unlearned": unlearned_results,
    }
    output_path = config.SAVES_DIR / "eval_results.json"
    config.SAVES_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print_comparison(original_results, unlearned_results)


if __name__ == "__main__":
    main()
