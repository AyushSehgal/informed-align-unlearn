from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Tuple, Optional
import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def load_pre_trained_llm(
    model_name: str,
    tokenizer_name: str,
    revision: str = "main",
    torch_dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = True,
    tokenizer_variant: str = "phi",
    **kwargs,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM + tokenizer with config-driven dtype, MoE/remote-code, and device_map.

    Extra kwargs beyond what transformers understands are silently ignored so
    yaml files can carry metadata like `tokenizer_variant`.
    """
    load_kwargs = {"revision": revision}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = _DTYPE_MAP[torch_dtype]
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer_kwargs = {"revision": revision, "padding_side": "left"}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Stash the variant so downstream data loaders can pick the right positive_*.json
    model.config.tokenizer_variant = tokenizer_variant
    return model, tokenizer

def load_pre_trained_text_embedding_model(
    model_name: str, tokenizer_name: str, **kwargs
) -> Tuple[AutoModel, AutoTokenizer]:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer
