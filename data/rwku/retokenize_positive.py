#!/usr/bin/env python
"""
Produce positive_{variant}.json files for RWKU targets so the align-then-
unlearn pipeline can load training text sized to a specific tokenizer.

The existing positive_phi.json entries are just `{"text": ...}` dicts — the
filename suggests Phi-specific tokenization but only raw text is stored. We
preserve that format and optionally drop entries whose token count under the
target tokenizer exceeds the requested max length (useful for models with
shorter effective context like Phi-3).

Usage:
    python data/rwku/retokenize_positive.py \\
        --tokenizer Qwen/Qwen3.5-4B \\
        --variant qwen \\
        --max_tokens 1024

This writes `positive_qwen.json` into each
`data/rwku/benchmark/Target/<target_id>/` directory alongside the existing
`positive_phi.json`.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def retokenize_target(
    target_dir: Path,
    tokenizer,
    variant: str,
    max_tokens: int | None,
    overwrite: bool,
) -> tuple[int, int]:
    src = target_dir / "positive_phi.json"
    dst = target_dir / f"positive_{variant}.json"
    if not src.exists():
        return (0, 0)
    if dst.exists() and not overwrite:
        print(f"  skip {target_dir.name}: {dst.name} exists (use --overwrite)")
        return (0, 0)

    with open(src) as f:
        entries = json.load(f)

    kept = []
    for entry in entries:
        text = entry.get("text")
        if not text:
            continue
        if max_tokens is not None:
            # Use tokenizer to check length, but don't store the ids — the
            # data pipeline tokenizes on the fly.
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_tokens:
                continue
        kept.append({"text": text})

    with open(dst, "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    return (len(kept), len(entries))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", required=True,
        help="HuggingFace tokenizer ID, e.g. Qwen/Qwen3.5-4B",
    )
    parser.add_argument(
        "--variant", required=True,
        help="Suffix used in the output filename, e.g. 'qwen'",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Drop entries that tokenize to more than this many tokens "
             "(default: keep everything)",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="data/rwku/benchmark/Target",
        help="Path to RWKU Target directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing positive_{variant}.json files",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"{data_root} not found; did you run download_rwku_data.sh?"
        )

    total_kept, total_seen = 0, 0
    targets = sorted(p for p in data_root.iterdir() if p.is_dir())
    print(f"Retokenizing {len(targets)} targets for variant={args.variant}")
    for target_dir in targets:
        kept, seen = retokenize_target(
            target_dir, tokenizer, args.variant, args.max_tokens, args.overwrite
        )
        if seen > 0:
            print(f"  {target_dir.name}: kept {kept}/{seen}")
        total_kept += kept
        total_seen += seen

    print(f"\nDone. Kept {total_kept}/{total_seen} entries across all targets.")
    if args.max_tokens is None:
        print("(no length filtering — output is a verbatim copy of positive_phi.json)")


if __name__ == "__main__":
    main()
