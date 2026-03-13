"""Step 7: Scaling experiment — measure unlearning effectiveness vs corpus size.

Runs the full pipeline at different corpus sizes to demonstrate that
the "Who is Harry Potter" approach requires massive corpus coverage.

Sizes tested: [128, 1000, 5000, 10000, 50000, 90000, 128000]

For each size:
  1. Build corpus of that size (RWKU base + model-generated + oversampled)
  2. Create sanitized counterparts (anchor-based replacement)
  3. Train reinforced model
  4. Generate alternative labels
  5. Train unlearned model
  6. Evaluate on RWKU
  7. Save results

Output: data/scaling_results.json + data/scaling_curve.png
"""

import os
import gc
import json
import random
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names

import config
from utils import load_tokenizer, load_base_model, read_json, write_json, chunk_text
import importlib
_anchors_mod = importlib.import_module("2_anchors")
sanitize_text = _anchors_mod.sanitize_text
get_sorted_anchors = _anchors_mod.get_sorted_anchors

# ── Experiment configuration ──────────────────────────────────────────────────
EXPERIMENT_SIZES = [128, 1000, 5000, 10000, 50000, 90000, 128000]

# How many unique passages to generate before oversampling
MAX_UNIQUE_GENERATIONS = 5000  # ~2-3 hrs on A100; beyond this we oversample

# Templates for generating diverse passages about the target
GENERATION_TEMPLATES = [
    "Write a detailed paragraph about {target} and their career.",
    "Describe the early life of {target}.",
    "Summarize the major achievements of {target}.",
    "Write about {target}'s most famous works.",
    "Describe the themes commonly found in {target}'s writing.",
    "Write about {target}'s influence on popular culture.",
    "Describe adaptations of {target}'s work into film.",
    "Write about awards received by {target}.",
    "Describe {target}'s writing style.",
    "Write about the characters created by {target}.",
    "Discuss the critical reception of {target}'s work.",
    "Write about {target}'s personal life and family.",
    "Describe controversies involving {target}.",
    "Compare {target} to other authors in their genre.",
    "Write about the publishing history of {target}'s books.",
    "Describe {target}'s impact on the horror genre.",
    "Write a biographical sketch of {target}.",
    "Discuss recurring motifs in {target}'s body of work.",
    "Write about places associated with {target}.",
    "Describe how {target}'s work evolved over the decades.",
    "Write about {target}'s early struggles as a writer.",
    "Describe the most iconic scenes from {target}'s novels.",
    "Write about the fan community around {target}'s work.",
    "Discuss the relationship between {target} and Hollywood.",
    "Write about {target}'s views on the craft of writing.",
    "Describe {target}'s collaboration with other authors.",
    "Write about {target}'s pen names and the works published under them.",
    "Discuss the supernatural elements in {target}'s fiction.",
    "Write about {target}'s memoir and non-fiction work.",
    "Describe the small-town settings in {target}'s novels.",
]


def free_gpu_memory():
    """Free GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


GENERATION_BATCH_SIZE = 8  # batch size for passage generation

def generate_text_batch(model, tokenizer, prompts, max_new_tokens=300, temperature=0.7):
    """Generate text from multiple prompts in a single batched call."""
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    results = []
    for i in range(len(prompts)):
        new_tokens = outputs[i][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(result)
    return results


def try_load_rwku_negatives():
    """Try to load RWKU negative/alternative passages for the target."""
    print("Exploring RWKU dataset for negative passages...")
    try:
        configs = get_dataset_config_names(config.RWKU_DATASET)
        print(f"Available RWKU configs ({len(configs)}): {configs[:20]}...")

        negative_configs = [c for c in configs if "negative" in c.lower() or "alter" in c.lower()]
        print(f"Potential negative configs: {negative_configs}")

        negative_passages = []
        for cfg in negative_configs:
            try:
                for split in ["train", "test"]:
                    try:
                        ds = load_dataset(config.RWKU_DATASET, cfg, split=split)
                        if "subject" in ds.column_names:
                            rows = [r for r in ds if r["subject"] == config.TARGET_NAME]
                        else:
                            rows = list(ds)
                        for row in rows:
                            text = row.get("text", row.get("passage", row.get("content", "")))
                            if isinstance(text, str) and len(text.strip()) > 50:
                                negative_passages.append(text.strip())
                    except Exception:
                        continue
            except Exception as e:
                print(f"  Could not load {cfg}: {e}")

        print(f"Found {len(negative_passages)} negative passages from RWKU")
        return negative_passages
    except Exception as e:
        print(f"Could not explore RWKU configs: {e}")
        return []


def build_corpus_at_size(target_size, base_corpus, base_sanitized, model, tokenizer, anchors, generated_pool=None):
    """Build a corpus of (original, sanitized) pairs at the target size.

    Strategy:
    - Start with base RWKU corpus (128 passages)
    - Add model-generated passages (up to MAX_UNIQUE_GENERATIONS)
    - For sizes beyond unique pool, oversample with shuffling
    """
    corpus = list(base_corpus)
    sanitized = list(base_sanitized)

    if target_size <= len(corpus):
        # Just subsample
        return corpus[:target_size], sanitized[:target_size]

    # Use pre-generated pool if available
    if generated_pool is not None:
        pool_orig, pool_san = generated_pool
        needed = target_size - len(corpus)
        available = len(pool_orig)

        if needed <= available:
            corpus.extend(pool_orig[:needed])
            sanitized.extend(pool_san[:needed])
        else:
            # Use all unique + oversample
            corpus.extend(pool_orig)
            sanitized.extend(pool_san)
            remaining = target_size - len(corpus)
            # Oversample from full pool (base + generated)
            full_orig = list(base_corpus) + list(pool_orig)
            full_san = list(base_sanitized) + list(pool_san)
            for i in range(remaining):
                idx = i % len(full_orig)
                corpus.append(full_orig[idx])
                sanitized.append(full_san[idx])

    assert len(corpus) == target_size, f"Expected {target_size}, got {len(corpus)}"
    assert len(sanitized) == target_size
    return corpus, sanitized


def generate_passage_pool(model, tokenizer, anchors, num_passages):
    """Generate a pool of diverse (original, sanitized) passage pairs using batched generation."""
    print(f"Generating pool of {num_passages} unique passages (batch_size={GENERATION_BATCH_SIZE})...")
    pool_orig = []
    pool_san = []

    templates = GENERATION_TEMPLATES
    remaining = num_passages

    pbar = tqdm(total=num_passages, desc="Generating passage pool")
    while remaining > 0:
        # Build a batch of prompts
        batch_size = min(GENERATION_BATCH_SIZE, remaining)
        prompts = []
        for _ in range(batch_size):
            template = random.choice(templates)
            prompts.append(template.format(target=config.TARGET_NAME))

        temp = 0.6 + random.random() * 0.4
        passages = generate_text_batch(model, tokenizer, prompts, max_new_tokens=400, temperature=temp)

        for passage in passages:
            if len(passage.split()) >= 20:
                san_passage = sanitize_text(passage, anchors)
                pool_orig.append(passage)
                pool_san.append(san_passage)

        pbar.update(batch_size)
        remaining -= batch_size

    pbar.close()
    print(f"Generated {len(pool_orig)} unique passages")
    return pool_orig, pool_san


def train_reinforced_for_experiment(corpus, tokenizer, experiment_dir):
    """Train reinforced model on given corpus. Returns the model."""
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_linear_schedule_with_warmup

    class TextChunkDataset(Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, idx):
            return torch.tensor(self.chunks[idx], dtype=torch.long)

    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    model = load_base_model(device_map=None)
    model.cuda()
    model.gradient_checkpointing_enable()

    all_chunks = []
    for text in corpus:
        all_chunks.extend(chunk_text(text, tokenizer, config.REINFORCED_CTX_LEN))
    print(f"  Reinforced training: {len(all_chunks)} chunks from {len(corpus)} passages")

    dataset = TextChunkDataset(all_chunks)
    dataloader = DataLoader(dataset, batch_size=config.REINFORCED_BATCH, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.REINFORCED_LR, weight_decay=0.01)
    total_steps = (len(dataloader) // config.REINFORCED_GRAD_ACCUM) * config.REINFORCED_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    model.train()
    device = next(model.parameters()).device

    for epoch in range(config.REINFORCED_EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"  Reinf epoch {epoch+1}/{config.REINFORCED_EPOCHS}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.REINFORCED_GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * config.REINFORCED_GRAD_ACCUM
            if (step + 1) % config.REINFORCED_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item() * config.REINFORCED_GRAD_ACCUM:.4f}")
        if (step + 1) % config.REINFORCED_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"  Reinf epoch {epoch+1} — avg loss: {total_loss / len(dataloader):.4f}")

    # Save
    reinforced_dir = experiment_dir / "reinforced"
    reinforced_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(reinforced_dir))
    tokenizer.save_pretrained(str(reinforced_dir))
    return model


def generate_labels_for_experiment(corpus, sanitized, tokenizer, baseline, reinforced, experiment_dir):
    """Generate alternative labels for given corpus."""
    import torch.nn.functional as F

    device = next(baseline.parameters()).device
    all_records = []

    for idx in tqdm(range(len(corpus)), desc="  Generating labels", leave=False):
        orig_chunks = chunk_text(corpus[idx], tokenizer, config.REINFORCED_CTX_LEN)
        san_chunks = chunk_text(sanitized[idx], tokenizer, config.REINFORCED_CTX_LEN)
        n = min(len(orig_chunks), len(san_chunks))

        for i in range(n):
            orig_ids = torch.tensor([orig_chunks[i]], device=device)
            san_ids = torch.tensor([san_chunks[i]], device=device)

            with torch.no_grad():
                baseline_logits = baseline(input_ids=san_ids).logits[0]
                reinforced_logits = reinforced(input_ids=orig_ids).logits[0]

            min_len = min(baseline_logits.size(0), reinforced_logits.size(0))
            bl = baseline_logits[:min_len]
            rl = reinforced_logits[:min_len]

            combined = bl.clone()
            mask = rl > bl
            combined[mask] = bl[mask] - config.ALPHA * F.relu(rl[mask] - bl[mask])

            soft_labels = F.softmax(combined, dim=-1)
            topk = 32
            topk_vals, topk_ids = soft_labels.topk(topk, dim=-1)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

            all_records.append({
                "passage_idx": idx,
                "chunk_idx": i,
                "input_ids": orig_chunks[i],
                "topk_ids": topk_ids.cpu().tolist(),
                "topk_vals": topk_vals.cpu().float().tolist(),
            })

    labels_path = experiment_dir / "alternative_labels.json"
    write_json(all_records, labels_path)
    print(f"  Generated {len(all_records)} label records")
    return all_records


def train_unlearn_for_experiment(records, tokenizer, experiment_dir):
    """Train unlearned model on alternative labels."""
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_linear_schedule_with_warmup

    class AlternativeLabelDataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            rec = self.records[idx]
            return (torch.tensor(rec["input_ids"], dtype=torch.long),
                    torch.tensor(rec["topk_ids"], dtype=torch.long),
                    torch.tensor(rec["topk_vals"], dtype=torch.float))

    def collate_fn(batch):
        input_ids_list, topk_ids_list, topk_vals_list = zip(*batch)
        max_len = max(x.size(0) for x in input_ids_list)
        topk = topk_ids_list[0].size(-1)
        bs = len(batch)
        input_ids = torch.zeros(bs, max_len, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
        label_ids = torch.zeros(bs, max_len, topk, dtype=torch.long)
        label_vals = torch.zeros(bs, max_len, topk, dtype=torch.float)
        for i, (ids, tids, tvals) in enumerate(batch):
            seq_len = ids.size(0)
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            lbl_len = tids.size(0)
            label_ids[i, :lbl_len] = tids
            label_vals[i, :lbl_len] = tvals
        return input_ids, attention_mask, label_ids, label_vals

    def soft_cross_entropy(logits, topk_ids, topk_vals, mask):
        logits = logits[:, :-1].contiguous()
        topk_ids = topk_ids[:, :-1].contiguous()
        topk_vals = topk_vals[:, :-1].contiguous()
        mask = mask[:, :-1].contiguous()
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=topk_ids)
        token_loss = -(gathered * topk_vals).sum(dim=-1)
        token_loss = token_loss * mask.float()
        return token_loss.sum() / mask.float().sum().clamp(min=1.0)

    model = load_base_model(device_map=None)
    model.cuda()
    model.gradient_checkpointing_enable()

    dataset = AlternativeLabelDataset(records)
    dataloader = DataLoader(dataset, batch_size=config.UNLEARN_BATCH, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.UNLEARN_LR, weight_decay=0.01)
    total_steps = (len(dataloader) // config.UNLEARN_GRAD_ACCUM) * config.UNLEARN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    model.train()
    device = next(model.parameters()).device

    for epoch in range(config.UNLEARN_EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"  Unlearn epoch {epoch+1}/{config.UNLEARN_EPOCHS}", leave=False)
        for step, (input_ids, attention_mask, label_ids, label_vals) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids = label_ids.to(device)
            label_vals = label_vals.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = soft_cross_entropy(outputs.logits, label_ids, label_vals, attention_mask)
            loss = loss / config.UNLEARN_GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * config.UNLEARN_GRAD_ACCUM
            if (step + 1) % config.UNLEARN_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item() * config.UNLEARN_GRAD_ACCUM:.4f}")
        if (step + 1) % config.UNLEARN_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"  Unlearn epoch {epoch+1} — avg loss: {total_loss / len(dataloader):.4f}")

    unlearned_dir = experiment_dir / "unlearned"
    unlearned_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(unlearned_dir))
    tokenizer.save_pretrained(str(unlearned_dir))
    return model


def evaluate_for_experiment(model, tokenizer, label="Model"):
    """Run RWKU evaluation (imported from 6_evaluate_rwku.py)."""
    from importlib import import_module
    eval_module = import_module("6_evaluate_rwku")
    return eval_module.evaluate_model(model, tokenizer, label)


def plot_scaling_curve(results, output_path):
    """Plot the scaling curve: corpus size vs forget ROUGE-L."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sizes = sorted(results.keys())
        forget_l1 = [results[s]["unlearned"]["forget"].get("level1", {}).get("rouge_l", 0) for s in sizes]
        forget_l2 = [results[s]["unlearned"]["forget"].get("level2", {}).get("rouge_l", 0) for s in sizes]
        forget_l3 = [results[s]["unlearned"]["forget"].get("level3", {}).get("rouge_l", 0) for s in sizes]
        orig_l1 = [results[s]["original"]["forget"].get("level1", {}).get("rouge_l", 0) for s in sizes]
        orig_l2 = [results[s]["original"]["forget"].get("level2", {}).get("rouge_l", 0) for s in sizes]
        orig_l3 = [results[s]["original"]["forget"].get("level3", {}).get("rouge_l", 0) for s in sizes]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, ul, ol, level in zip(axes, [forget_l1, forget_l2, forget_l3],
                                       [orig_l1, orig_l2, orig_l3],
                                       ["Level 1 (Fill-in-blank)", "Level 2 (QA)", "Level 3 (Adversarial)"]):
            ax.plot(sizes, ul, 'b-o', label='Unlearned', linewidth=2, markersize=8)
            ax.plot(sizes, ol, 'r--s', label='Original (baseline)', linewidth=2, markersize=8, alpha=0.7)
            ax.set_xlabel('Corpus Size (# passages)', fontsize=12)
            ax.set_ylabel('Forget ROUGE-L (lower = better)', fontsize=12)
            ax.set_title(f'Forget Set — {level}', fontsize=13)
            ax.set_xscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Scaling Curve: Corpus Size vs Unlearning Effectiveness\n'
                      '("Who is Harry Potter" approach on Stephen King / Qwen 2.5 3B)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scaling curve saved to {output_path}")

        # Also plot MIA and utility
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        mia_forget = [results[s]["unlearned"]["mia"]["forget"]["avg_loss"] for s in sizes]
        mia_orig = [results[s]["original"]["mia"]["forget"]["avg_loss"] for s in sizes]
        ax1.plot(sizes, mia_forget, 'b-o', label='Unlearned', linewidth=2, markersize=8)
        ax1.plot(sizes, mia_orig, 'r--s', label='Original', linewidth=2, markersize=8, alpha=0.7)
        ax1.set_xlabel('Corpus Size')
        ax1.set_ylabel('MIA Forget Loss (higher = better)')
        ax1.set_title('MIA Resistance')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        utility = [results[s]["unlearned"]["utility"]["avg_loss"] for s in sizes]
        util_orig = [results[s]["original"]["utility"]["avg_loss"] for s in sizes]
        ax2.plot(sizes, utility, 'b-o', label='Unlearned', linewidth=2, markersize=8)
        ax2.plot(sizes, util_orig, 'r--s', label='Original', linewidth=2, markersize=8, alpha=0.7)
        ax2.set_xlabel('Corpus Size')
        ax2.set_ylabel('Utility Loss (lower = better)')
        ax2.set_title('General Utility Preservation')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path2 = str(output_path).replace('.png', '_mia_utility.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"MIA/Utility plot saved to {output_path2}")

    except ImportError:
        print("matplotlib not available, skipping plot generation")


def run_single_experiment(size, corpus, sanitized, tokenizer, baseline, anchors, experiment_dir):
    """Run full pipeline for one corpus size. Returns eval results dict."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: corpus_size = {size}")
    print(f"{'='*70}")

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save corpus for this experiment
    write_json(corpus, experiment_dir / "forget_corpus.json")
    write_json(sanitized, experiment_dir / "sanitized_corpus.json")

    # Step 1: Train reinforced model
    print(f"\n[1/4] Training reinforced model on {len(corpus)} passages...")
    t0 = time.time()
    reinforced = train_reinforced_for_experiment(corpus, tokenizer, experiment_dir)
    print(f"  Reinforced training took {(time.time()-t0)/60:.1f} min")

    # Step 2: Generate alternative labels
    print(f"\n[2/4] Generating alternative labels...")
    reinforced.eval()
    baseline.eval()
    t0 = time.time()
    records = generate_labels_for_experiment(corpus, sanitized, tokenizer, baseline, reinforced, experiment_dir)
    print(f"  Label generation took {(time.time()-t0)/60:.1f} min")

    # Free reinforced model
    del reinforced
    free_gpu_memory()

    # Step 3: Train unlearned model
    print(f"\n[3/4] Training unlearned model...")
    t0 = time.time()
    unlearned = train_unlearn_for_experiment(records, tokenizer, experiment_dir)
    print(f"  Unlearn training took {(time.time()-t0)/60:.1f} min")

    # Step 4: Evaluate
    print(f"\n[4/4] Evaluating...")
    t0 = time.time()

    # Evaluate original (baseline)
    original_results = evaluate_for_experiment(baseline, tokenizer, f"Original (size={size})")

    # Evaluate unlearned
    unlearned.eval()
    unlearned_results = evaluate_for_experiment(unlearned, tokenizer, f"Unlearned (size={size})")
    print(f"  Evaluation took {(time.time()-t0)/60:.1f} min")

    # Free unlearned model
    del unlearned
    free_gpu_memory()

    result = {
        "original": original_results,
        "unlearned": unlearned_results,
        "corpus_size": size,
        "num_unique_passages": min(size, len(corpus)),
    }

    # Save individual result
    write_json(result, experiment_dir / "eval_results.json")
    return result


def main():
    print("=" * 70)
    print("SCALING EXPERIMENT")
    print(f"Sizes to test: {EXPERIMENT_SIZES}")
    print("=" * 70)

    tokenizer = load_tokenizer()
    anchors = get_sorted_anchors()

    # Load base corpus
    print("\nLoading base RWKU corpus...")
    base_corpus = read_json(config.DATA_DIR / "forget_corpus.json")
    base_sanitized = read_json(config.DATA_DIR / "sanitized_corpus.json")
    print(f"Base corpus: {len(base_corpus)} passages")

    # Try to get RWKU negatives
    rwku_negatives = try_load_rwku_negatives()
    if rwku_negatives:
        write_json(rwku_negatives, config.DATA_DIR / "rwku_negatives.json")

    # Check for existing scaling results (to resume)
    scaling_results_path = config.DATA_DIR / "scaling_results.json"
    if scaling_results_path.exists():
        all_results = {int(k): v for k, v in read_json(scaling_results_path).items()}
        print(f"Resuming: found existing results for sizes {list(all_results.keys())}")
    else:
        all_results = {}

    # Determine max unique passages needed
    max_size = max(EXPERIMENT_SIZES)
    unique_needed = min(max_size - len(base_corpus), MAX_UNIQUE_GENERATIONS)

    # Generate passage pool (only once, reused across experiments)
    pool_path = config.DATA_DIR / "generated_pool.json"
    claude_pool_path = config.DATA_DIR / "generated_pool_claude.json"
    if pool_path.exists():
        print("Loading existing generated pool...")
        pool_data = read_json(pool_path)
        pool_orig = pool_data["original"]
        pool_san = pool_data["sanitized"]
        print(f"Loaded pool: {len(pool_orig)} passages")
    elif claude_pool_path.exists():
        print("Loading pre-generated Claude pool...")
        pool_data = read_json(claude_pool_path)
        pool_orig = pool_data["original"]
        pool_san = pool_data["sanitized"]
        # Save as standard pool so future runs find it
        write_json(pool_data, pool_path)
        print(f"Loaded Claude pool: {len(pool_orig)} passages")
    else:
        print(f"\nGenerating passage pool ({unique_needed} passages)...")
        model = load_base_model()
        model.eval()
        pool_orig, pool_san = generate_passage_pool(model, tokenizer, anchors, unique_needed)
        del model
        free_gpu_memory()

        # Save pool for reuse
        write_json({"original": pool_orig, "sanitized": pool_san}, pool_path)

    # Load baseline model (kept in memory for evaluation)
    print("\nLoading baseline model (kept for all evaluations)...")
    baseline = load_base_model()
    baseline.eval()

    # Run experiments
    for size in EXPERIMENT_SIZES:
        if size in all_results:
            print(f"\nSkipping size={size} (already have results)")
            continue

        # Build corpus at this size
        corpus, sanitized = build_corpus_at_size(
            size, base_corpus, base_sanitized,
            baseline, tokenizer, anchors,
            generated_pool=(pool_orig, pool_san)
        )

        experiment_dir = config.SAVES_DIR / f"scaling_{size}"
        result = run_single_experiment(size, corpus, sanitized, tokenizer, baseline, anchors, experiment_dir)
        all_results[size] = result

        # Save after each experiment (in case of crash)
        write_json({str(k): v for k, v in all_results.items()}, scaling_results_path)
        print(f"\nSaved results so far to {scaling_results_path}")

    # Final save
    write_json({str(k): v for k, v in all_results.items()}, scaling_results_path)

    # Plot
    plot_scaling_curve(all_results, config.DATA_DIR / "scaling_curve.png")

    # Print summary
    print(f"\n{'='*70}")
    print("SCALING EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':>8} | {'Forget L1':>10} | {'Forget L2':>10} | {'Forget L3':>10} | {'Orig L3':>10}")
    print("-" * 60)
    for size in sorted(all_results.keys()):
        r = all_results[size]
        fl1 = r["unlearned"]["forget"].get("level1", {}).get("rouge_l", 0)
        fl2 = r["unlearned"]["forget"].get("level2", {}).get("rouge_l", 0)
        fl3 = r["unlearned"]["forget"].get("level3", {}).get("rouge_l", 0)
        ol3 = r["original"]["forget"].get("level3", {}).get("rouge_l", 0)
        print(f"{size:>8} | {fl1:>10.4f} | {fl2:>10.4f} | {fl3:>10.4f} | {ol3:>10.4f}")


if __name__ == "__main__":
    main()
