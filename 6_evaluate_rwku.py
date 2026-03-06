"""Step 6: Evaluate original and unlearned models on the RWKU benchmark.

RWKU evaluation subsets for a target entity:
  - Forget set: fill-in-blank, QA, adversarial (ROUGE-L, lower = better unlearning)
  - Neighbor set: related-entity QA (ROUGE-L, higher = better preservation)
  - MIA set: membership inference attack (LOSS metric)
  - Utility set: general benchmarks (MMLU, TruthfulQA, etc.)
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer

import config
from utils import load_tokenizer, load_base_model, load_unlearned_model, read_json


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
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the generated part
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

    # Shift for causal LM loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return loss.item()


def eval_forget_set(model, tokenizer, target_dir):
    """Evaluate on forget set (fill-in-blank, QA, adversarial)."""
    results = {}

    for task in ["fill_in_blank", "question_answering", "adversarial"]:
        task_path = target_dir / "forget" / f"{task}.json"
        if not task_path.exists():
            print(f"  Skipping {task} (file not found: {task_path})")
            continue

        data = read_json(task_path)
        scores = []
        for item in tqdm(data, desc=f"  Forget/{task}", leave=False):
            prompt = item.get("question", item.get("prompt", ""))
            reference = item.get("answer", item.get("ground_truth", ""))
            if not prompt or not reference:
                continue
            prediction = generate_response(model, tokenizer, prompt)
            score = compute_rouge_l(prediction, reference)
            scores.append(score)

        if scores:
            results[task] = {
                "rouge_l": np.mean(scores),
                "n": len(scores),
            }
            print(f"  Forget/{task}: ROUGE-L = {np.mean(scores):.4f} (n={len(scores)})")

    return results


def eval_neighbor_set(model, tokenizer, target_dir):
    """Evaluate on neighbor set (related entity preservation)."""
    neighbor_path = target_dir / "neighbor" / "question_answering.json"
    if not neighbor_path.exists():
        # Try alternative path structures
        neighbor_dir = target_dir / "neighbor"
        if neighbor_dir.exists():
            files = list(neighbor_dir.glob("*.json"))
            if files:
                neighbor_path = files[0]
            else:
                print("  Skipping neighbor set (no files found)")
                return {}
        else:
            print(f"  Skipping neighbor set (dir not found)")
            return {}

    data = read_json(neighbor_path)
    scores = []
    for item in tqdm(data, desc="  Neighbor", leave=False):
        prompt = item.get("question", item.get("prompt", ""))
        reference = item.get("answer", item.get("ground_truth", ""))
        if not prompt or not reference:
            continue
        prediction = generate_response(model, tokenizer, prompt)
        score = compute_rouge_l(prediction, reference)
        scores.append(score)

    result = {}
    if scores:
        result["rouge_l"] = np.mean(scores)
        result["n"] = len(scores)
        print(f"  Neighbor: ROUGE-L = {np.mean(scores):.4f} (n={len(scores)})")
    return result


def eval_mia_set(model, tokenizer, target_dir):
    """Evaluate membership inference attack resistance."""
    results = {}
    for split in ["member", "nonmember"]:
        mia_path = target_dir / "mia" / f"{split}.json"
        if not mia_path.exists():
            continue

        data = read_json(mia_path)
        losses = []
        for item in tqdm(data, desc=f"  MIA/{split}", leave=False):
            text = item if isinstance(item, str) else item.get("text", "")
            if text:
                loss = compute_loss(model, tokenizer, text)
                losses.append(loss)

        if losses:
            results[split] = {
                "avg_loss": np.mean(losses),
                "n": len(losses),
            }
            print(f"  MIA/{split}: avg_loss = {np.mean(losses):.4f} (n={len(losses)})")

    return results


def eval_utility(model, tokenizer, target_dir):
    """Evaluate general utility (simple fluency/perplexity check).

    Full MMLU/BBH/TruthfulQA evaluation requires separate benchmark harnesses.
    Here we do a basic fluency check on general prompts.
    """
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

    result = {"avg_loss": np.mean(losses), "n": len(losses)}
    print(f"  Utility: avg_loss = {np.mean(losses):.4f}")
    return result


def evaluate_model(model, tokenizer, label="Model"):
    """Run all RWKU evaluations on a model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    target_dir = config.TARGET_DIR
    results = {}

    print("\n[Forget Set] (lower ROUGE-L = better unlearning)")
    results["forget"] = eval_forget_set(model, tokenizer, target_dir)

    print("\n[Neighbor Set] (higher ROUGE-L = better preservation)")
    results["neighbor"] = eval_neighbor_set(model, tokenizer, target_dir)

    print("\n[MIA Set] (member loss should be high = forgot the data)")
    results["mia"] = eval_mia_set(model, tokenizer, target_dir)

    print("\n[Utility] (lower loss = model still useful)")
    results["utility"] = eval_utility(model, tokenizer, target_dir)

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
        ("Forget/fill_in_blank ROUGE-L ↓", ["forget", "fill_in_blank", "rouge_l"]),
        ("Forget/QA ROUGE-L ↓", ["forget", "question_answering", "rouge_l"]),
        ("Forget/adversarial ROUGE-L ↓", ["forget", "adversarial", "rouge_l"]),
        ("Neighbor ROUGE-L ↑", ["neighbor", "rouge_l"]),
        ("MIA/member loss ↑", ["mia", "member", "avg_loss"]),
        ("MIA/nonmember loss", ["mia", "nonmember", "avg_loss"]),
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
