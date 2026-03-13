#!/usr/bin/env python3
"""
LUNE Baseline Experiment — Phi-3.5-mini-instruct on RWKU
Refactored from lune.ipynb for batch execution.

Reference: https://arxiv.org/pdf/2512.07375v1
"""

import os, json, random, copy, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Hyperparameters (Appendix B.1 / B.2)
# ============================================================================
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES_CANDIDATES = [
    # Mistral/LLaMA-style
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Phi-family variants
    "qkv_proj", "gate_up_proj",
]

# Training config
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 1024
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 40

# Early stopping
GUR_TOLERANCE = 0.005

# Paths
DATA_DIR = Path.home() / ".cache" / "huggingface" / "datasets" / "RWKU"
BATCH_DIR = DATA_DIR / "Batch" / "1-5"
TARGET_DIR = DATA_DIR / "Target"
FORGET_TARGET = "1_Stephen_King"
EXPERIMENT_DATA_DIR = TARGET_DIR / FORGET_TARGET
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. Dataset
# ============================================================================
class NegativeExampleDataset(Dataset):
    """Dataset of (prompt, negative_response) pairs for LUNE unlearning."""

    def __init__(self, tokenizer, data_dir, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self._load_negatives(data_dir)
        print(f"Loaded {len(self.examples)} negative examples")

    def _load_negatives(self, data_dir):
        neg_path = data_dir / "negative.json"
        if neg_path.exists():
            negatives = json.load(open(neg_path))
            self._process_negatives(negatives)
        else:
            for target_dir in sorted(TARGET_DIR.iterdir()):
                neg_file = target_dir / "negative.json"
                if neg_file.exists():
                    negatives = json.load(open(neg_file))
                    self._process_negatives(negatives)

    def _process_negatives(self, negatives):
        uncertainty_phrases = [
            "i am not sure", "i'm not sure", "it might be",
            "i don't know", "i'm uncertain", "possibly", "perhaps",
            "it could be", "maybe", "not certain",
        ]
        for item in negatives:
            text = item["text"]
            subject = item.get("subject", "")
            intro = item.get("intro", "")
            if any(phrase in text.lower() for phrase in uncertainty_phrases):
                continue
            prompt = f"Tell me about {subject}." if intro else f"Who is {subject}?"
            self.examples.append({
                "prompt": prompt,
                "negative_response": text,
                "subject": subject,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text = ex["prompt"]
        response_text = ex["negative_response"]
        full_text = f"{prompt_text} {response_text}"

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=True, truncation=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def balance_negatives_per_subject(dataset, max_per_subject=150):
    subject_counts = {}
    balanced = []
    for ex in dataset.examples:
        subj = ex["subject"]
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
        if subject_counts[subj] <= max_per_subject:
            balanced.append(ex)
    dataset.examples = balanced
    print(f"After balancing: {len(dataset.examples)} examples "
          f"across {len(subject_counts)} subjects")
    return dataset


def resolve_lora_target_modules(model):
    linear_module_suffixes = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    resolved = [
        module_name
        for module_name in LORA_TARGET_MODULES_CANDIDATES
        if module_name in linear_module_suffixes
    ]
    if not resolved:
        raise ValueError(
            "Could not infer LoRA target modules for this model. "
            f"Tried: {LORA_TARGET_MODULES_CANDIDATES}"
        )
    return resolved


# ============================================================================
# 2. Loss (Eq. 4)
# ============================================================================
def compute_negative_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"].to(DEVICE),
        attention_mask=batch["attention_mask"].to(DEVICE),
        labels=batch["labels"].to(DEVICE),
    )
    return outputs.loss


# ============================================================================
# 4. Evaluation metrics (Section 4.2)
# ============================================================================
def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_usr(model, tokenizer, data_dir, level="level1"):
    forget_file = data_dir / f"forget_{level}.json"
    if not forget_file.exists():
        return None
    forget_data = json.load(open(forget_file))
    success = sum(
        1 for item in forget_data
        if item["answer"].lower() not in generate_response(model, tokenizer, item["query"]).lower()
    )
    return success / len(forget_data) if forget_data else 0.0


def compute_gur_acc(model, tokenizer, data_dir):
    neighbor_file = data_dir / "neighbor_level1.json"
    if not neighbor_file.exists():
        return None
    neighbor_data = json.load(open(neighbor_file))
    correct = sum(
        1 for item in neighbor_data
        if item["answer"].lower() in generate_response(model, tokenizer, item["query"]).lower()
    )
    return correct / len(neighbor_data) if neighbor_data else 0.0


def compute_apr(model, tokenizer, data_dir):
    scores = []
    for level in ["level2", "level3"]:
        score = compute_usr(model, tokenizer, data_dir, level=level)
        if score is not None:
            scores.append(score)
    return np.mean(scores) if scores else None


def compute_mia(model, tokenizer, data_dir):
    forget_mia_file = data_dir / "forget_mia.json"
    retain_mia_file = data_dir / "retain_mia.json"
    if not forget_mia_file.exists() or not retain_mia_file.exists():
        return None

    forget_mia = json.load(open(forget_mia_file))
    retain_mia = json.load(open(retain_mia_file))

    def get_perplexity(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        return outputs.loss.item()

    member_ppls = [get_perplexity(item["text"]) for item in forget_mia]
    nonmember_ppls = [get_perplexity(item["text"]) for item in retain_mia]
    all_ppls = member_ppls + nonmember_ppls
    labels = [1] * len(member_ppls) + [0] * len(nonmember_ppls)
    threshold = np.median(all_ppls)
    predictions = [1 if p < threshold else 0 for p in all_ppls]
    return accuracy_score(labels, predictions)

def actual_gur(acc, baseline_acc=None):
    """Compute GUR from accuracy and length of neighbor set."""
    if baseline_acc is None:
        return acc
    return acc / baseline_acc if baseline_acc > 0 else 0.0


def evaluate_all(model, tokenizer, data_dir, baseline_gur=None):
    model.eval()
    gur_acc = compute_gur_acc(model, tokenizer, data_dir)
    return {
        "USR": compute_usr(model, tokenizer, data_dir),
        "GUR_acc": gur_acc,
        "GUR": actual_gur(gur_acc, baseline_gur),
        "APR": compute_apr(model, tokenizer, data_dir),
        "MIA": compute_mia(model, tokenizer, data_dir),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Forgetting target: {FORGET_TARGET}")
    print(f"Experiment data dir: {EXPERIMENT_DATA_DIR}")
    if not EXPERIMENT_DATA_DIR.exists():
        raise FileNotFoundError(f"Target directory not found: {EXPERIMENT_DATA_DIR}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_free, gpu_total = torch.cuda.mem_get_info()
        print(f"GPU memory — total: {gpu_total / (1024**3):.1f} GB, "
              f"free: {gpu_free / (1024**3):.1f} GB")

    # --- Load model & tokenizer ---
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Dataset ---
    train_dataset = NegativeExampleDataset(tokenizer, EXPERIMENT_DATA_DIR)
    train_dataset = balance_negatives_per_subject(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    # --- LoRA ---
    lora_target_modules = resolve_lora_target_modules(model)
    print(f"LoRA target modules: {lora_target_modules}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=lora_target_modules,
        bias="none", init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # --- Experiment directory ---
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUT_DIR / f"experiment_{experiment_id}"
    experiment_dir.mkdir(exist_ok=True, parents=True)
    print(f"Experiment dir: {experiment_dir}")

    # --- Baseline evaluation ---
    print("\nEvaluating baseline (before unlearning) ...")
    baseline_results = evaluate_all(model, tokenizer, EXPERIMENT_DATA_DIR)
    print(f"Baseline: {baseline_results}")
    baseline_gur = baseline_results.get("GUR", 1.0)
    json.dump(baseline_results, open(experiment_dir / "baseline_results.json", "w"), indent=2)

    # --- Training loop (Algorithm 1) ---
    print(f"\nStarting LUNE training for {NUM_EPOCHS} epochs ...")
    model.train()
    best_usr, best_epoch = 0.0, 0
    training_log = []
    
    # Create metrics file with headers
    metrics_file = experiment_dir / "metrics_by_epoch.csv"
    with open(metrics_file, "w") as f:
        f.write("epoch,loss,USR,GUR_acc,GUR,APR,MIA\n")

    for epoch in range(NUM_EPOCHS):
        epoch_loss, num_batches = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(pbar):
            loss = compute_negative_loss(model, batch) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluate every epoch
        print(f"\n--- Eval at epoch {epoch+1} ---")
        results = evaluate_all(model, tokenizer, EXPERIMENT_DATA_DIR, baseline_gur)
        results["epoch"] = epoch + 1
        results["loss"] = avg_loss
        training_log.append(results)
        
        # Save to JSON log
        json.dump(training_log, open(experiment_dir / "training_log.json", "w"), indent=2)
        
        # Save to CSV file
        with open(metrics_file, "a") as f:
            usr = results.get("USR", "N/A")
            gur_acc = results.get("GUR_acc", "N/A")
            gur = results.get("GUR", "N/A")
            apr = results.get("APR", "N/A")
            mia = results.get("MIA", "N/A")
            f.write(f"{epoch+1},{avg_loss},{usr},{gur_acc},{gur},{apr},{mia}\n")

        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Track best model (no early stopping)
        # current_gur = results.get("GUR", 0)
        # current_usr = results.get("USR", 0)
        # if current_gur is not None and current_usr is not None:
        #     gur_drop = (baseline_gur - current_gur) if baseline_gur else 0
        #     if gur_drop <= GUR_TOLERANCE and current_usr > best_usr:
        #         best_usr = current_usr
        #         best_epoch = epoch + 1
        #         model.save_pretrained(experiment_dir / "best_lora_adapter")
        #         tokenizer.save_pretrained(experiment_dir / "best_lora_adapter")
        #         print(f"  -> New best! USR={current_usr:.4f}, GUR drop={gur_drop:.4f}")
        #         json.dump({**results, "best_epoch": best_epoch, "best_usr": best_usr},
        #                   open(experiment_dir / "best_results.json", "w"), indent=2)

        model.train()

    print(f"\nTraining complete. Best USR: {best_usr:.4f} at epoch {best_epoch}")

    # --- Save final checkpoint & summary ---
    model.save_pretrained(experiment_dir / "final_lora_adapter")
    tokenizer.save_pretrained(experiment_dir / "final_lora_adapter")

    summary = {
        "experiment_id": experiment_id,
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "completed_epochs": epoch + 1,
        "best_epoch": best_epoch,
        "best_usr": best_usr,
        # "baseline_results": baseline_results,
        "final_training_log": training_log,
        "hyperparameters": {
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_seq_len": MAX_SEQ_LEN, "gur_tolerance": GUR_TOLERANCE,
        },
    }
    json.dump(summary, open(experiment_dir / "experiment_summary.json", "w"), indent=2)
    print(f"All results saved to {experiment_dir}")

    # --- Final evaluation on best checkpoint ---
    best_adapter_path = experiment_dir / "best_lora_adapter"
    if best_adapter_path.exists():
        print("\nLoading best checkpoint for final eval ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        best_model = PeftModel.from_pretrained(base_model, best_adapter_path)
        best_model.eval()
        final_results = evaluate_all(best_model, tokenizer, EXPERIMENT_DATA_DIR, baseline_gur)
    else:
        model.eval()
        final_results = evaluate_all(model, tokenizer, EXPERIMENT_DATA_DIR, baseline_gur)

    paper_results = {"USR": 88.5, "GUR": 93.7, "APR": 79.4, "MIA": 18.8}
    print("\n" + "=" * 60)
    print(f"{'Metric':<10} {'Ours':>10} {'Paper (LUNE)':>15}")
    print("=" * 60)
    for metric in ["USR", "GUR", "APR", "MIA"]:
        ours = final_results.get(metric)
        ours_str = f"{ours * 100:.1f}%" if ours is not None else "N/A"
        print(f"{metric:<10} {ours_str:>10} {paper_results[metric]:>14.1f}%")
    print("=" * 60)

    json.dump({"final_results": final_results, "paper_results": paper_results},
              open(experiment_dir / "final_comparison.json", "w"), indent=2)

    # --- Qualitative examples ---
    forget_file = EXPERIMENT_DATA_DIR / "forget_level1.json"
    if forget_file.exists():
        eval_model = best_model if best_adapter_path.exists() else model
        orig_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        forget_data = json.load(open(forget_file))[:5]

        qual_path = experiment_dir / "qualitative_examples.txt"
        with open(qual_path, "w") as f:
            f.write("=" * 70 + "\nQUALITATIVE COMPARISON: Before vs After Unlearning\n" + "=" * 70 + "\n")
            for item in forget_data:
                query, answer = item["query"], item["answer"]
                orig_resp = generate_response(orig_model, tokenizer, query)
                unlearn_resp = generate_response(eval_model, tokenizer, query)
                block = (f"\nPrompt: {query}\nTarget (should forget): {answer}\n"
                         f"BEFORE: {orig_resp}\nAFTER:  {unlearn_resp}\n" + "-" * 70 + "\n")
                f.write(block)
                print(block[:300])

        del orig_model
        torch.cuda.empty_cache()
        print(f"Qualitative examples saved to {qual_path}")


if __name__ == "__main__":
    main()
