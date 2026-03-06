"""Step 3: Fine-tune a 'reinforced' model on the forget corpus.

The reinforced model becomes an expert on J.K. Rowling content so that its
logits can later be compared to the baseline to identify JKR-specific tokens.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import config
from utils import load_tokenizer, load_base_model, read_json, chunk_text


class TextChunkDataset(Dataset):
    """Dataset of fixed-length token chunks from the forget corpus."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def train():
    tokenizer = load_tokenizer()
    model = load_base_model()
    model.gradient_checkpointing_enable()

    # Load forget corpus
    corpus_path = config.DATA_DIR / "forget_corpus.json"
    corpus = read_json(corpus_path)
    print(f"Loaded {len(corpus)} passages from forget corpus")

    # Tokenize into fixed-length chunks
    all_chunks = []
    for text in corpus:
        all_chunks.extend(chunk_text(text, tokenizer, config.REINFORCED_CTX_LEN))
    print(f"Created {len(all_chunks)} chunks of ≤{config.REINFORCED_CTX_LEN} tokens")

    dataset = TextChunkDataset(all_chunks)
    dataloader = DataLoader(
        dataset,
        batch_size=config.REINFORCED_BATCH,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.REINFORCED_LR, weight_decay=0.01
    )
    total_steps = (len(dataloader) // config.REINFORCED_GRAD_ACCUM) * config.REINFORCED_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    model.train()
    device = next(model.parameters()).device

    for epoch in range(config.REINFORCED_EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.REINFORCED_EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Standard causal LM loss: predict next token
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # ignore padding

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config.REINFORCED_GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * config.REINFORCED_GRAD_ACCUM

            if (step + 1) % config.REINFORCED_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * config.REINFORCED_GRAD_ACCUM:.4f}")

        # Flush any remaining accumulated gradients at end of epoch
        if (step + 1) % config.REINFORCED_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    # Save full model
    config.REINFORCED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.REINFORCED_DIR))
    tokenizer.save_pretrained(str(config.REINFORCED_DIR))
    print(f"Reinforced model saved to {config.REINFORCED_DIR}")


if __name__ == "__main__":
    train()
