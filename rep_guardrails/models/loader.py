"""
Model loading and architecture inspection utilities for autoregressive LLMs
and TransformerLens hooked models.
"""

from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from rep_guardrails.config import MODEL_REGISTRY, DEFAULT_MODEL_KEY, get_device, get_default_dtype


def show_architecture(model: HookedTransformer, name: Optional[str] = None) -> None:
    """
    Displays architectural specifications and dimensions of a HookedTransformer model.
    
    Args:
        model: Loaded HookedTransformer instance.
        name: Optional human-readable name for logging.
    """
    name_str = name or getattr(model.cfg, "model_name", "Hooked Model")
    print(f"\n{'='*20} {name_str} Architecture {'='*20}")
    print(f"Number of layers:     {model.cfg.n_layers}")
    print(f"Hidden dimension:     {model.cfg.d_model}")
    print(f"Attention heads:      {model.cfg.n_heads}")
    print(f"Head dimension:       {model.cfg.d_head}")
    print(f"MLP dimension:        {model.cfg.d_mlp}")
    print(f"Vocabulary size:      {model.cfg.d_vocab}")
    print(f"Context length:       {model.cfg.n_ctx}")
    print(f"Device:               {model.cfg.device}")
    print(f"Dtype:                {model.cfg.dtype}")
    print(f"{'='*60}\n")


def load_model_bundle(
    model_key: str = DEFAULT_MODEL_KEY,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
) -> Tuple[HookedTransformer, AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads native HuggingFace weights and wraps them in a HookedTransformer
    for mechanistic interpretability and activation extraction.
    
    This decoupled bundle enables:
      1. Fast native C++ generation via the HuggingFace AutoModelForCausalLM.
      2. Hooked non-autoregressive verification via TransformerLens without weight duplication.
    
    Args:
        model_key: Identifier from MODEL_REGISTRY (e.g. 'llama3-8b-instruct').
        device: PyTorch device ('cuda', 'mps', 'cpu'). Auto-detected if None.
        dtype: PyTorch precision. Auto-detected if None.
        hf_token: Optional Hugging Face access token for gated models.
        
    Returns:
        Tuple of (hooked_model, hf_model, tokenizer).
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Supported models: {list(MODEL_REGISTRY.keys())}"
        )

    device = device or get_device()
    dtype = dtype or get_default_dtype(device)
    cfg = MODEL_REGISTRY[model_key]
    hf_path = cfg["hf_path"]
    tl_template = cfg["tl_template"]

    print(f"[ModelLoader] Loading native HF weights for '{model_key}' from '{hf_path}' ({dtype} on {device})...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=dtype,
        token=hf_token,
    ).to(device)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_path, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[ModelLoader] Wrapping weights as HookedTransformer using template '{tl_template}'...")
    hooked_model = HookedTransformer.from_pretrained_no_processing(
        tl_template,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    hooked_model.eval()

    print("[ModelLoader] Model bundle loaded and initialized successfully.")
    return hooked_model, hf_model, tokenizer


def load_model_lite(
    name: str,
    hf_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
) -> HookedTransformer:
    """
    Directly loads a HookedTransformer model without caching separate native HF references.
    
    Args:
        name: Name for logging.
        hf_path: HuggingFace model repo id or local directory.
        device: Compute device.
        dtype: Data type precision.
        hf_token: Optional Hugging Face token.
        
    Returns:
        HookedTransformer instance.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype(device)
    print(f"[ModelLoader] Loading HookedTransformer directly: {name} ({hf_path})...")
    model = HookedTransformer.from_pretrained(
        hf_path,
        device=device,
        dtype=dtype,
        token=hf_token,
    )
    model.eval()
    return model
