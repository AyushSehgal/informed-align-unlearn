"""Step 1: Download RWKU data from HuggingFace and build the forget corpus."""

from datasets import load_dataset

import config
from utils import write_json


def build_forget_corpus():
    """Load RWKU training passages and filter for the target subject."""
    print(f"Loading RWKU training passages for '{config.TARGET_NAME}'...")
    ds = load_dataset(config.RWKU_DATASET, "train_original_passage", split="train")

    # Filter for target subject
    target_rows = [row for row in ds if row["subject"] == config.TARGET_NAME]
    print(f"Found {len(target_rows)} passages for {config.TARGET_NAME}")

    if not target_rows:
        # Show available subjects to help debug
        subjects = sorted(set(row["subject"] for row in ds))
        print(f"Available subjects ({len(subjects)}):")
        for s in subjects[:20]:
            print(f"  {s}")
        raise ValueError(f"Target '{config.TARGET_NAME}' not found in RWKU dataset")

    # Extract passage text
    corpus_texts = []
    for row in target_rows:
        text = row.get("text", row.get("passage", ""))
        if isinstance(text, str) and text.strip():
            corpus_texts.append(text.strip())

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.DATA_DIR / "forget_corpus.json"
    write_json(corpus_texts, output_path)
    print(f"Saved {len(corpus_texts)} passages to {output_path}")

    # Also save as plain text for inspection
    txt_path = config.DATA_DIR / "forget_corpus.txt"
    txt_path.write_text("\n\n---\n\n".join(corpus_texts), encoding="utf-8")
    print(f"Saved plain text corpus to {txt_path}")

    return corpus_texts


if __name__ == "__main__":
    corpus = build_forget_corpus()
    print(f"\nDone! Forget corpus has {len(corpus)} passages.")
