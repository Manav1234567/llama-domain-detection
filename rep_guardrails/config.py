"""
Configuration parameters, model registries, and constants for the
Representation Engineering Guardrails framework.
"""

from typing import Dict, List, Any
import os
import torch

# ==============================================================================
# Model Architecture Registry
# ==============================================================================
# Defines supported causal language models, their Hugging Face repository IDs,
# and their corresponding TransformerLens template configurations.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "llama3-8b-instruct": {
        "hf_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tl_template": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "mistral-7b-instruct-v0.2": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "tl_template": "mistralai/Mistral-7B-Instruct-v0.1",
    },
    "mistral-7b-instruct-v0.3": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "tl_template": "mistralai/Mistral-7B-Instruct-v0.1",
    },
}

DEFAULT_MODEL_KEY = "llama3-8b-instruct"

# ==============================================================================
# Feature Extraction Keys
# ==============================================================================
# Multidimensional tensor metrics collected across all transformer layers
FEATURE_KEYS: List[str] = [
    "residual_means",      # Sequence-averaged hidden state vector [n_layers, d_model]
    "residual_last",       # Terminal token hidden state vector [n_layers, d_model]
    "mlp_first",           # Query encoding state at token 0 (MHAD proxy) [n_layers, d_mlp]
    "mlp_last",            # Terminal factual retrieval state (MHAD proxy) [n_layers, d_mlp]
    "attn_entropy_mean",   # Average attention entropy across query tokens [n_layers, n_heads]
    "attn_entropy_max",    # Peak attention entropy across query tokens [n_layers, n_heads]
    "lookback_ratio_mean", # Mean attention ratio (context vs generation) [n_layers, n_heads]
    "lookback_ratio_last", # Terminal token lookback attention ratio [n_layers, n_heads]
]

# Zero-marginal-cost token logit metrics extracted during verification forward pass
SCALAR_KEYS: List[str] = [
    "output_entropy_mean",    # Mean output logit entropy over generated sequence
    "output_entropy_max",     # Peak output logit entropy over generated sequence
    "top1_prob_mean",         # Average top-1 softmax probability
    "top1_top2_margin_mean",  # Average confidence gap between top-1 and top-2 logits
]

# ==============================================================================
# Ground Truth Judge & Evaluation Settings
# ==============================================================================
DEFAULT_JUDGE_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
FALLBACK_JUDGE_MODEL = "facebook/bart-large-mnli"

# Severity threshold for binary domain classification (> 0.1 indicates positive target domain)
SEVERITY_THRESHOLD: float = 0.1

# Outlier filtering confidence level using Chi-Square distribution on squared standardized distances
OUTLIER_CONFIDENCE: float = 0.995

# PCA dimensionality reduction components for probe pipelines
N_PCA_COMPONENTS: int = 50

# Default file paths
DEFAULT_DATASET_PATH = "political_probe_dataset.pkl"
DEFAULT_CLASSIFIER_PATH = "political_flag_classifier.pkl"
DEFAULT_PILOT_PATH = "domain_severity_pilot.pkl"

# Compliance override message for inference guardrail
DEFAULT_INTERCEPTION_MESSAGE = (
    "I'm sorry, but I am not authorized to discuss political topics."
)

# ==============================================================================
# Hardware / Device Utilities
# ==============================================================================
def get_device(preferred: str = None) -> torch.device:
    """
    Detects and returns the optimal available PyTorch compute device.
    Prefers user selection, then Apple Silicon MPS, NVIDIA CUDA, and defaults to CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_dtype(device: torch.device) -> torch.dtype:
    """
    Determines optimal floating point precision for the active hardware.
    Uses bfloat16 for modern CUDA, float16 for MPS, and float32 for CPU.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.float16
    return torch.float32
