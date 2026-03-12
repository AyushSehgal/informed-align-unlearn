"""Step 1b: Augment the forget corpus using the model itself.

The original 128 RWKU passages are too few to effectively unlearn.
This script prompts the base model to generate diverse content about
the target subject, then generates sanitized counterparts.

This mirrors the original "Who is Harry Potter" paper's approach of
using a large, comprehensive corpus covering all facets of the target's knowledge.
"""

import json
import torch
from tqdm import tqdm

import config
from utils import load_tokenizer, load_base_model, read_json, write_json

# Diverse prompt templates to cover all facets of the target's knowledge
PROMPT_TEMPLATES = [
    # Biographical
    "Write a detailed biographical paragraph about {target}.",
    "Describe the early life and childhood of {target}.",
    "Summarize the career and major achievements of {target}.",
    "Describe the personal life and family of {target}.",
    "Write about {target}'s influence and legacy.",

    # Works (for authors/artists — adapt if target is different)
    "Summarize the plot of a famous work by {target}.",
    "Describe the major themes in the works of {target}.",
    "Write about the characters created by {target}.",
    "Discuss the writing style and techniques of {target}.",
    "List and briefly describe the most important works by {target}.",

    # Cultural impact
    "Describe how {target} influenced popular culture.",
    "Write about adaptations of {target}'s work into other media.",
    "Discuss the critical reception of {target}'s most famous work.",
    "Describe awards and honors received by {target}.",
    "Write about controversies or criticism involving {target}.",

    # Relationships and context
    "Describe {target}'s relationships with other notable figures in the same field.",
    "Compare {target}'s style with that of their contemporaries.",
    "Write about the places associated with {target}.",
    "Describe how {target}'s background influenced their work.",
    "Write a timeline of key events in {target}'s life and career.",

    # Detailed knowledge
    "Describe a specific scene or passage from a work by {target}.",
    "Write about recurring motifs in {target}'s body of work.",
    "Discuss how {target}'s work evolved over time.",
    "Write about the publishing history of {target}'s works.",
    "Describe the fan community around {target}'s work.",

    # QA-style
    "What is {target} best known for? Provide a detailed answer.",
    "What are the main genres {target} works in?",
    "Where was {target} born and raised?",
    "What was {target}'s first major success?",
    "How has {target}'s work been adapted for film or television?",
]

# Generic replacement name for sanitization
GENERIC_NAME = "Robert James Miller"
GENERIC_DESCRIPTOR = "the author"

SANITIZE_PROMPT = """Rewrite the following passage, replacing all references to {target} with "{generic_name}" (a fictional author). Replace all specific works, characters, places, and details unique to {target} with plausible but generic fictional alternatives. Keep the same structure and length. Do NOT mention {target} at all.

Original passage:
{passage}

Rewritten passage:"""


def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7):
    """Generate text from a prompt using the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def augment_corpus():
    tokenizer = load_tokenizer()
    print("Loading base model for augmentation...")
    model = load_base_model()
    model.eval()

    # Load existing corpus
    existing_corpus = read_json(config.DATA_DIR / "forget_corpus.json")
    existing_sanitized = read_json(config.DATA_DIR / "sanitized_corpus.json")
    print(f"Existing corpus: {len(existing_corpus)} passages")

    new_passages = []
    new_sanitized = []

    # Generate multiple variations per prompt template
    num_variations = 3  # generates len(PROMPT_TEMPLATES) * num_variations new passages
    total = len(PROMPT_TEMPLATES) * num_variations
    print(f"Generating {total} new passages...")

    pbar = tqdm(total=total, desc="Generating passages")
    for template in PROMPT_TEMPLATES:
        for v in range(num_variations):
            # Generate original passage about the target
            prompt = template.format(target=config.TARGET_NAME)
            passage = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=400,
                temperature=0.7 + (v * 0.1),  # vary temperature across variations
            )

            if len(passage.split()) < 20:
                pbar.update(1)
                continue  # skip very short generations

            new_passages.append(passage)

            # Generate sanitized version
            san_prompt = SANITIZE_PROMPT.format(
                target=config.TARGET_NAME,
                generic_name=GENERIC_NAME,
                passage=passage,
            )
            sanitized = generate_text(
                model, tokenizer, san_prompt,
                max_new_tokens=500,
                temperature=0.3,  # lower temp for more faithful rewriting
            )

            if len(sanitized.split()) < 20:
                # Fallback: simple string replacement
                sanitized = passage.replace(config.TARGET_NAME, GENERIC_NAME)

            new_sanitized.append(sanitized)
            pbar.update(1)

    pbar.close()
    print(f"Generated {len(new_passages)} new passages")

    # Combine with existing
    combined_corpus = existing_corpus + new_passages
    combined_sanitized = existing_sanitized + new_sanitized

    assert len(combined_corpus) == len(combined_sanitized), (
        f"Mismatch: {len(combined_corpus)} corpus vs {len(combined_sanitized)} sanitized"
    )

    # Save augmented versions
    write_json(combined_corpus, config.DATA_DIR / "forget_corpus.json")
    write_json(combined_sanitized, config.DATA_DIR / "sanitized_corpus.json")
    print(f"Saved augmented corpus: {len(combined_corpus)} total passages")

    # Also save just the new passages separately for inspection
    write_json(new_passages, config.DATA_DIR / "augmented_passages.json")
    write_json(new_sanitized, config.DATA_DIR / "augmented_sanitized.json")
    print(f"Saved new passages separately for inspection")


if __name__ == "__main__":
    augment_corpus()
