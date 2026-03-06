"""Step 5: Fine-tune the baseline model on alternative labels (the actual unlearning).

The model learns to predict the 'generic' label distribution instead of
JKR-specific tokens, effectively unlearning J.K. Rowling knowledge.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import config
from utils import load_tokenizer, load_base_model, read_json


class AlternativeLabelDataset(Dataset):
    """Dataset of (input_ids, soft_label_topk_ids, soft_label_topk_vals)."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        topk_ids = torch.tensor(rec["topk_ids"], dtype=torch.long)
        topk_vals = torch.tensor(rec["topk_vals"], dtype=torch.float)
        return input_ids, topk_ids, topk_vals


def collate_fn(batch):
    """Pad batch to same length."""
    input_ids_list, topk_ids_list, topk_vals_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)
    topk = topk_ids_list[0].size(-1)
    bs = len(batch)

    input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    # Labels have shape (seq_len, topk) — we pad the seq dimension
    label_ids = torch.zeros(bs, max_len, topk, dtype=torch.long)
    label_vals = torch.zeros(bs, max_len, topk, dtype=torch.float)

    for i, (ids, tids, tvals) in enumerate(batch):
        seq_len = ids.size(0)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        # topk_ids/vals have shape (seq_len, topk) — these are the labels
        # for positions 0..seq_len-1 (predicting the NEXT token at each position)
        lbl_len = tids.size(0)
        label_ids[i, :lbl_len] = tids
        label_vals[i, :lbl_len] = tvals

    return input_ids, attention_mask, label_ids, label_vals


def soft_cross_entropy(logits, topk_ids, topk_vals, mask):
    """Cross-entropy between model logits and sparse soft labels.

    Args:
        logits: (batch, seq, vocab)
        topk_ids: (batch, seq, topk) — token indices
        topk_vals: (batch, seq, topk) — probabilities (sum to 1)
        mask: (batch, seq) — 1 for real tokens, 0 for padding
    """
    # Shift: logits[t] predicts token[t+1], labels[t] are the target for position t
    # So logits[:, :-1] should match labels[:, 1:]
    logits = logits[:, :-1].contiguous()
    topk_ids = topk_ids[:, 1:].contiguous()
    topk_vals = topk_vals[:, 1:].contiguous()
    mask = mask[:, 1:].contiguous()

    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq-1, vocab)

    # Gather log probs at the topk positions
    # topk_ids: (batch, seq-1, topk)
    gathered = log_probs.gather(dim=-1, index=topk_ids)  # (batch, seq-1, topk)

    # Weighted sum: sum_k p_k * log(q_k)
    token_loss = -(gathered * topk_vals).sum(dim=-1)  # (batch, seq-1)

    # Mask and average
    token_loss = token_loss * mask.float()
    loss = token_loss.sum() / mask.float().sum().clamp(min=1.0)
    return loss


def train():
    tokenizer = load_tokenizer()
    model = load_base_model()
    model.gradient_checkpointing_enable()

    # Load alternative labels
    records = read_json(config.DATA_DIR / "alternative_labels.json")
    print(f"Loaded {len(records)} label records")

    dataset = AlternativeLabelDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.UNLEARN_LR, weight_decay=0.01
    )
    total_steps = (len(dataloader) // config.UNLEARN_GRAD_ACCUM) * config.UNLEARN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    model.train()
    device = next(model.parameters()).device

    for epoch in range(config.UNLEARN_EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.UNLEARN_EPOCHS}")
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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    # Save full model
    config.UNLEARNED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.UNLEARNED_DIR))
    tokenizer.save_pretrained(str(config.UNLEARNED_DIR))
    print(f"Unlearned model saved to {config.UNLEARNED_DIR}")


if __name__ == "__main__":
    train()
