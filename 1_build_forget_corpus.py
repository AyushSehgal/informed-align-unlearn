"""Step 1: Download RWKU data and build the forget corpus for J.K. Rowling."""

import json
import zipfile
import subprocess
import sys
from pathlib import Path

import config
from utils import read_json, write_json


def download_rwku():
    """Download and extract RWKU dataset from Google Drive."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.DATA_DIR / "RWKU.zip"

    if config.RWKU_DIR.exists():
        print("RWKU data already exists, skipping download.")
        return

    print("Downloading RWKU dataset...")
    subprocess.check_call([
        sys.executable, "-m", "gdown",
        f"https://drive.google.com/uc?id={config.RWKU_GDRIVE_ID}",
        "-O", str(zip_path),
    ])

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.DATA_DIR)

    zip_path.unlink()
    print(f"RWKU extracted to {config.RWKU_DIR}")


def build_forget_corpus():
    """Build the forget corpus from RWKU passage data."""
    passage_path = config.TARGET_DIR / "passage.json"
    if not passage_path.exists():
        raise FileNotFoundError(
            f"Passage file not found at {passage_path}. "
            "Did the RWKU download succeed?"
        )

    passages = read_json(passage_path)
    print(f"Loaded {len(passages)} passages from {passage_path}")

    # Build a single corpus text from all passages
    corpus_texts = []
    for entry in passages:
        # passage.json entries may be dicts with a "text" field or just strings
        if isinstance(entry, dict):
            text = entry.get("text", entry.get("passage", ""))
        else:
            text = str(entry)
        text = text.strip()
        if text:
            corpus_texts.append(text)

    output_path = config.DATA_DIR / "forget_corpus.json"
    write_json(corpus_texts, output_path)
    print(f"Saved {len(corpus_texts)} passages to {output_path}")

    # Also save as a single text file for easy inspection
    txt_path = config.DATA_DIR / "forget_corpus.txt"
    txt_path.write_text("\n\n---\n\n".join(corpus_texts), encoding="utf-8")
    print(f"Saved plain text corpus to {txt_path}")

    return corpus_texts


if __name__ == "__main__":
    download_rwku()
    corpus = build_forget_corpus()
    print(f"\nDone! Forget corpus has {len(corpus)} passages.")
