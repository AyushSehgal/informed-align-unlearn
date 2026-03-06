"""Shared utilities for the unlearning pipeline."""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_tokenizer():
    """Load the tokenizer for the target model."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(device_map="auto", torch_dtype=torch.bfloat16):
    """Load the base (pre-trained) model."""
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def load_reinforced_model(device_map="auto", torch_dtype=torch.bfloat16):
    """Load the saved reinforced model (full fine-tuned)."""
    model = AutoModelForCausalLM.from_pretrained(
        str(config.REINFORCED_DIR),
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def load_unlearned_model(device_map="auto", torch_dtype=torch.bfloat16):
    """Load the saved unlearned model (full fine-tuned)."""
    model = AutoModelForCausalLM.from_pretrained(
        str(config.UNLEARNED_DIR),
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def read_json(path):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """Write data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chunk_text(text, tokenizer, max_len):
    """Split text into chunks of max_len tokens, returning token id lists."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_len):
        chunk = token_ids[i : i + max_len]
        if len(chunk) > 10:  # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks
