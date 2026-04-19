from pathlib import Path

import lightning.pytorch as pl
import torch
from project.utils.logging import get_logger
import json
import re
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
from itertools import cycle



log = get_logger()

data_root = Path(__file__).parent.parent / "data/rwku/benchmark/Target"


_LAST_NAME_MIN_LEN = 5


def _derive_aliases(target_name: str, extra_aliases: Optional[List[str]]) -> List[str]:
    """Build the alias list used to localize subject tokens.

    Always includes the full name and possessive variants. Bare last name
    is added only when it's >= `_LAST_NAME_MIN_LEN` chars, to avoid common
    surnames like "Smith" or "Lee" creating false-positive matches across
    the retain text. Extra aliases from `<target_dir>/aliases.json` are
    unioned in without length filtering — the user opted into them.
    Longer strings first so the regex prefers the most-specific span.
    """
    names = {target_name.strip()}
    parts = target_name.strip().split()
    if len(parts) > 1 and len(parts[-1]) >= _LAST_NAME_MIN_LEN:
        names.add(parts[-1])
    if extra_aliases:
        for a in extra_aliases:
            a = (a or "").strip()
            if a:
                names.add(a)
    expanded = set()
    for n in names:
        expanded.add(n)
        expanded.add(f"{n}'s")
        expanded.add(f"{n}\u2019s")
    return sorted(expanded, key=len, reverse=True)


def _load_aliases(target_dir: Path) -> List[str]:
    """Load optional aliases from `<target_dir>/aliases.json` (a list of strings
    or a dict with an "aliases" key). Missing or unparseable files return []."""
    path = target_dir / "aliases.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load aliases for {target_dir.name}: {e}")
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, str)]
    if isinstance(data, dict):
        raw = data.get("aliases") or data.get("names") or []
        return [x for x in raw if isinstance(x, str)]
    return []


class RWKUPositiveDataset(Dataset):
    def __init__(
        self,
        target_id: str,
        max_input_length: int,
        context_window_length: int,
        primary_tokenizer: AutoTokenizer,
        secondary_tokenizer: AutoTokenizer | None = None,
        tokenizer_variant: str = "phi",
        target_name: Optional[str] = None,
        subject_mask_window: int = 0,
    ):
        primary_tokenizer.pad_token = primary_tokenizer.eos_token
        if secondary_tokenizer is not None:
            secondary_tokenizer.pad_token = secondary_tokenizer.eos_token
        self.primary_tokenizer = primary_tokenizer
        self.secondary_tokenizer = secondary_tokenizer
        self.max_input_length = max_input_length
        self.context_window_length = context_window_length
        self.subject_mask_window = max(0, int(subject_mask_window))

        # target_name falls back to the canonical "1_Stephen_King" -> "Stephen King"
        # convention used in train.py. Aliases are unioned from aliases.json if present.
        if target_name is None:
            target_name = target_id.split("_", 1)[1].replace("_", " ") if "_" in target_id else target_id
        self.target_name = target_name
        target_dir = data_root / target_id
        extra = _load_aliases(target_dir)
        self._alias_list = _derive_aliases(target_name, extra)
        # Pre-compile one alternation pattern (longest-first, case-insensitive).
        self._alias_pattern = re.compile(
            "|".join(re.escape(a) for a in self._alias_list),
            flags=re.IGNORECASE,
        )

        # Pick the positive_{variant}.json file. The file stores raw text,
        # so for new tokenizers we can fall back to positive_phi.json and
        # retokenize on the fly.
        candidate_files = [
            data_root / target_id / f"positive_{tokenizer_variant}.json",
            data_root / target_id / "positive_phi.json",
        ]
        positive_path = None
        for candidate in candidate_files:
            if candidate.exists():
                positive_path = candidate
                break
        if positive_path is None:
            raise FileNotFoundError(
                f"No positive data found for {target_id}. Tried: "
                f"{[str(c) for c in candidate_files]}"
            )

        self.texts = []
        with open(positive_path, "r") as f:
            entries = json.load(f)
            for entry in entries:
                self.texts.append(entry["text"])

        assert len(self.texts) > 0, f"No positive texts found for target {target_id}"
        log.info(
            f"Loaded {len(self.texts)} positive texts for {target_id} "
            f"from {positive_path.name}"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._process_text(self.texts[index])

    def _build_subject_mask(
        self,
        text: str,
        primary_offsets: List,
        length: int,
    ) -> torch.Tensor:
        """Token-level mask: 1 where the primary token's character offsets
        overlap any alias match in `text`. Dilated by `subject_mask_window`.

        Uses offset_mapping instead of token-ID subsequence matching so we
        catch BPE leading-space variants, possessives, and case differences.
        """
        mask = torch.zeros(length, dtype=torch.long)
        spans = [(m.start(), m.end()) for m in self._alias_pattern.finditer(text)]
        if not spans:
            return mask

        for i in range(length):
            if i >= len(primary_offsets):
                break
            off = primary_offsets[i]
            # Special/pad tokens have offset (0, 0) on HF fast tokenizers.
            if off is None:
                continue
            ostart, oend = int(off[0]), int(off[1])
            if oend <= ostart:
                continue
            for sstart, send in spans:
                if ostart < send and oend > sstart:
                    mask[i] = 1
                    break

        if self.subject_mask_window > 0 and mask.any():
            # Dilate by a 1D max-pool of width (2w+1).
            w = self.subject_mask_window
            dilated = torch.nn.functional.max_pool1d(
                mask.float().view(1, 1, -1),
                kernel_size=2 * w + 1,
                stride=1,
                padding=w,
            ).view(-1).long()
            mask = dilated[:length]
        return mask

    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        self.primary_tokenizer_output = self.primary_tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=self.max_input_length + 1,
            padding="max_length",
            padding_side="right",
            truncation=True,
        )
        primary_tokens = self.primary_tokenizer_output["input_ids"]
        primary_offsets = self.primary_tokenizer_output["offset_mapping"]

        primary_input_ids = primary_tokens[: self.max_input_length]
        primary_labels = primary_tokens[1 : self.max_input_length + 1]

        # primary_input_ids is a Python list, so `list != int` returns a
        # single bool instead of an element-wise comparison — yielding a
        # 0-dim scalar mask that collates into a 1D [batch] tensor and
        # crashes newer transformers masking (Qwen3.5). Tensor-ify first.
        primary_ids_tensor = torch.tensor(primary_input_ids)
        attention_mask_tensor = (
            primary_ids_tensor != self.primary_tokenizer.pad_token_id
        ).long()

        subject_mask_tensor = self._build_subject_mask(
            text, primary_offsets, self.max_input_length
        )
        # Zero out any mask hits on pad positions (defensive; offsets are (0,0) there).
        subject_mask_tensor = subject_mask_tensor * attention_mask_tensor

        if self.secondary_tokenizer is None:
            return {
                "primary_input_ids": primary_ids_tensor,
                "primary_labels": torch.tensor(primary_labels),
                "attention_mask": attention_mask_tensor,
                "subject_mask": subject_mask_tensor,
            }

        self.secondary_tokenizer_output = self.secondary_tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding_side="right",
        )
        secondary_tokens = self.secondary_tokenizer_output["input_ids"]
        secondary_offsets = self.secondary_tokenizer_output["offset_mapping"]

        secondary_context_windows = []

        for ptoken, (pstart, pend) in zip(
            primary_tokens[: self.max_input_length],
            primary_offsets[: self.max_input_length],
        ):
            if ptoken == self.primary_tokenizer.pad_token_id:
                secondary_context_windows.append(
                    torch.tensor(
                        [self.secondary_tokenizer.pad_token_id]
                        * self.context_window_length
                    )
                )
                continue

            context_window_for_token = []

            for stoken, (sstart, send) in zip(secondary_tokens, secondary_offsets):
                if stoken == self.secondary_tokenizer.pad_token_id:
                    break
                if send > pend:
                    context_window_for_token.append(stoken)
                if len(context_window_for_token) == self.context_window_length:
                    break

            context_window_for_token = context_window_for_token + [
                self.secondary_tokenizer.pad_token_id
            ] * (self.context_window_length - len(context_window_for_token))

            secondary_context_windows.append(torch.tensor(context_window_for_token))

        has_full_window = [
            (window != self.secondary_tokenizer.pad_token_id).all().item()
            for window in secondary_context_windows
        ]

        return {
            "primary_input_ids": primary_ids_tensor,
            "primary_labels": torch.tensor(primary_labels),
            "secondary_context_windows": torch.stack(secondary_context_windows),
            "attention_mask": attention_mask_tensor,
            "has_full_window": torch.tensor(has_full_window),
            "subject_mask": subject_mask_tensor,
        }

class RWKUPositiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target_ids: List[str],
        batch_size: int,
        num_workers: int,
        max_input_length: int,
        context_window_length: int,
        primary_tokenizer: AutoTokenizer,
        secondary_tokenizer: AutoTokenizer | None = None,
        tokenizer_variant: str = "phi",
        target_names: Optional[Dict[str, str]] = None,
        subject_mask_window: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.target_ids = target_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.primary_tokenizer = primary_tokenizer
        self.secondary_tokenizer = secondary_tokenizer
        self.max_input_length = max_input_length
        self.context_window_length = context_window_length
        self.tokenizer_variant = tokenizer_variant
        self.target_names = target_names or {}
        self.subject_mask_window = subject_mask_window
        self.datasets = [
            RWKUPositiveDataset(
                target_id=target_id,
                primary_tokenizer=self.primary_tokenizer,
                secondary_tokenizer=self.secondary_tokenizer,
                max_input_length=self.max_input_length,
                context_window_length=self.context_window_length,
                tokenizer_variant=self.tokenizer_variant,
                target_name=self.target_names.get(target_id),
                subject_mask_window=self.subject_mask_window,
            )
            for target_id in self.target_ids
        ]
        self.merged_dataset = ConcatDataset(self.datasets)

    def _infinite_loader(self, loader: DataLoader):
        while True:
            for batch in iter(loader):
                yield batch

    def train_dataloader(self):
        return self._infinite_loader(
            DataLoader(
                self.merged_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
        )


def main():
    primary_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct"
    )
    secondary_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    dataset = RWKUPositiveDataset(
        "1_Stephen_King", primary_tokenizer, secondary_tokenizer, 512, 3
    )

    def print_idx(item_idx: int, token_idx: int):
        item = dataset[item_idx]
        print("-" * 10 + f"Index: ({item_idx}, {token_idx})" + "-" * 10)
        print(primary_tokenizer.decode(item["primary_input_ids"][token_idx]))
        print(primary_tokenizer.decode(item["primary_labels"][token_idx]))
        print(secondary_tokenizer.decode(item["secondary_context_windows"][token_idx]))

    print_idx(0, 0)
    print_idx(0, 1)
    print_idx(0, 2)
    print_idx(0, 3)
    print_idx(0, 4)
    print_idx(0, 5)
    print_idx(0, len(dataset[0]["primary_input_ids"]) - 2)
    print_idx(0, len(dataset[0]["primary_input_ids"]) - 1)


if __name__ == "__main__":
    main()
