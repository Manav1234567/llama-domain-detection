"""
Mechanistic activation extraction and PyTorch hook management for autoregressive LLMs.

Implements the decoupled "Fast Generation, Hooked Verification" paradigm:
1. Native causal LM generates output tokens rapidly via optimized C++ kernels.
2. A single non-autoregressive hooked forward pass extracts internal representations
   across residual stream, attention matrices, and MLP intermediate states.
"""

from typing import Dict, List, Any, Optional, Union
import gc
import os
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from rep_guardrails.config import FEATURE_KEYS, SCALAR_KEYS


class PolygraphFeatureExtractor:
    """
    Hooked feature extractor for transformer architectures. Captures dimensional representations
    from the residual stream, attention distributions, MLP post-activations, and output logit dynamics.
    """

    def __init__(
        self,
        model: HookedTransformer,
        hf_gen_model: AutoModelForCausalLM,
        hf_tokenizer: AutoTokenizer,
        device: torch.device,
    ) -> None:
        """
        Initialize the feature extractor with dual model references.
        
        Args:
            model: HookedTransformer instance for hooked forward pass inspection.
            hf_gen_model: Native HuggingFace AutoModelForCausalLM for high-speed generation.
            hf_tokenizer: Tokenizer for text encoding/decoding.
            device: Active computation device.
        """
        self.model = model
        self.hf_gen_model = hf_gen_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.n_layers: int = model.cfg.n_layers
        self.n_heads: int = model.cfg.n_heads
        self.current_prompt_len: int = 0
        self.hooks = []
        self.reset_storage()

    def reset_storage(self) -> None:
        """Clears intermediate tensor storage buffers to prevent memory leaks across runs."""
        self.storage: Dict[str, Dict[int, np.ndarray]] = {
            "residual_means": {},
            "residual_last": {},
            "mlp_first": {},
            "mlp_last": {},
            "attn_entropy_mean": {},
            "attn_entropy_max": {},
            "lookback_ratio_mean": {},
            "lookback_ratio_last": {},
        }

    def register_hooks(self) -> None:
        """Attaches PyTorch forward hooks across all transformer layers."""
        self.hooks = []
        for layer in range(self.n_layers):
            # 1. Post-residual stream hook (ACT-ViT representation engineering)
            self.hooks.append(
                self.model.blocks[layer].hook_resid_post.add_hook(
                    lambda val, hook, l=layer: self._hook_residual(val, hook, l)
                )
            )
            # 2. Attention pattern hook (Attention entropy and Lookback ratio)
            self.hooks.append(
                self.model.blocks[layer].attn.hook_pattern.add_hook(
                    lambda val, hook, l=layer: self._hook_attention(val, hook, l)
                )
            )
            # 3. Post-MLP hook (MHAD query encoding and factual termination)
            self.hooks.append(
                self.model.blocks[layer].mlp.hook_post.add_hook(
                    lambda val, hook, l=layer: self._hook_mlp(val, hook, l)
                )
            )

    def remove_hooks(self) -> None:
        """Detaches all active forward hooks from the HookedTransformer."""
        self.model.reset_hooks()
        self.hooks = []

    def _hook_residual(self, value: torch.Tensor, hook: Any, layer_idx: int) -> torch.Tensor:
        """
        Extracts residual stream representations:
        - Sequence-averaged mean vector (captures holistic topic state across time)
        - Terminal token vector (captures final state right before unembedding)
        """
        tensor = value.detach().to(torch.float32).cpu()[0]  # Shape: [seq_len, d_model]
        self.storage["residual_means"][layer_idx] = tensor.mean(dim=0).numpy()
        self.storage["residual_last"][layer_idx] = tensor[-1, :].numpy()
        return value

    def _hook_mlp(self, value: torch.Tensor, hook: Any, layer_idx: int) -> torch.Tensor:
        """
        Extracts MLP intermediate activation vectors:
        - Token 0 (query encoding anchor per MHAD framework)
        - Token -1 (terminal factual retrieval anchor)
        """
        tensor = value.detach().to(torch.float32).cpu()[0]  # Shape: [seq_len, d_mlp]
        self.storage["mlp_first"][layer_idx] = tensor[0, :].numpy()
        self.storage["mlp_last"][layer_idx] = tensor[-1, :].numpy()
        return value

    def _hook_attention(self, value: torch.Tensor, hook: Any, layer_idx: int) -> torch.Tensor:
        """
        Extracts attention distribution metrics over generated tokens:
        - Attention entropy (measures uncertainty/cognitive dissonance)
        - Lookback ratio (ratio of attention directed at prompt vs newly generated tokens)
        """
        matrix = value.detach().to(torch.float32).cpu()[0]  # Shape: [heads, seq_q, seq_k]
        epsilon = 1e-9
        p_len = self.current_prompt_len

        # Analyze strictly the response query positions (decision points during generation)
        response_matrix = matrix[:, p_len:, :]  # Shape: [heads, resp_len, seq_k]
        
        # Calculate attention entropy per head across response positions
        entropy = -torch.sum(response_matrix * torch.log(response_matrix + epsilon), dim=-1)
        self.storage["attn_entropy_mean"][layer_idx] = entropy.mean(dim=-1).numpy()
        self.storage["attn_entropy_max"][layer_idx] = entropy.max(dim=-1).values.numpy()

        # Lookback ratio at the terminal generated token
        last_attn = matrix[:, -1, :]  # Shape: [heads, seq_k]
        prompt_attn = last_attn[:, :p_len].sum(dim=-1)
        gen_attn = last_attn[:, p_len:].sum(dim=-1)
        self.storage["lookback_ratio_last"][layer_idx] = (prompt_attn / (gen_attn + epsilon)).numpy()

        # Mean lookback ratio averaged across all response tokens
        resp_prompt = response_matrix[:, :, :p_len].sum(dim=-1)
        resp_gen = response_matrix[:, :, p_len:].sum(dim=-1)
        self.storage["lookback_ratio_mean"][layer_idx] = (
            (resp_prompt / (resp_gen + epsilon)).mean(dim=-1).numpy()
        )
        return value

    def _output_stats(self, logits: torch.Tensor, p_len: int) -> Dict[str, float]:
        """
        Computes token-level probability statistics with zero marginal cost
        from logits already produced by the hooked forward pass.
        
        Args:
            logits: Output tensor of shape [1, seq_len, vocab_size].
            p_len: Prompt token length.
            
        Returns:
            Dictionary with entropy and margin metrics.
        """
        # Response token logits (from token predicting the first response token to the end)
        response_logits = logits[0, p_len - 1 : -1, :].float()
        probs = torch.softmax(response_logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        return {
            "output_entropy_mean": entropy.mean().item(),
            "output_entropy_max": entropy.max().item(),
            "top1_prob_mean": top2[:, 0].mean().item(),
            "top1_top2_margin_mean": (top2[:, 0] - top2[:, 1]).mean().item(),
        }

    def generate_native(self, prompt_text: str, max_new_tokens: int = 50) -> str:
        """
        Executes fast autoregressive text generation using native C++ kernels.
        
        Args:
            prompt_text: Formatted prompt text (including chat template headers).
            max_new_tokens: Maximum tokens to generate.
            
        Returns:
            Decoded full string (prompt + generation).
        """
        inputs = self.hf_tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        with torch.inference_mode():
            out = self.hf_gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.hf_tokenizer.eos_token_id,
            )

        return self.hf_tokenizer.decode(out[0], skip_special_tokens=False)

    def extract_from_text(
        self,
        prompt_text: str,
        response_text: Optional[str] = None,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Performs generation (if response_text not provided) and a single hooked verification
        forward pass to extract mechanistic feature representations.
        
        Args:
            prompt_text: Full formatted prompt text.
            response_text: Pre-generated response text (optional).
            max_new_tokens: Tokens to generate if response_text is None.
            
        Returns:
            Dictionary of extracted feature arrays and scalar statistics.
        """
        if response_text is None:
            full_generation = self.generate_native(prompt_text, max_new_tokens=max_new_tokens)
            resp = full_generation[len(prompt_text):].strip()
        else:
            full_generation = prompt_text + response_text
            resp = response_text.strip()

        prompt_tokens = self.model.to_tokens(prompt_text)
        self.current_prompt_len = prompt_tokens.shape[1]

        self.reset_storage()
        self.register_hooks()
        with torch.inference_mode():
            logits = self.model(full_generation)
        self.remove_hooks()

        out_stats = self._output_stats(logits, self.current_prompt_len)
        del logits

        entry: Dict[str, Any] = {
            "prompt": prompt_text,
            "response": resp,
            "full_generation": full_generation,
            "residual_means": np.array([self.storage["residual_means"][l] for l in range(self.n_layers)]),
            "residual_last": np.array([self.storage["residual_last"][l] for l in range(self.n_layers)]),
            "mlp_first": np.array([self.storage["mlp_first"][l] for l in range(self.n_layers)]),
            "mlp_last": np.array([self.storage["mlp_last"][l] for l in range(self.n_layers)]),
            "attn_entropy_mean": np.array([self.storage["attn_entropy_mean"][l] for l in range(self.n_layers)]),
            "attn_entropy_max": np.array([self.storage["attn_entropy_max"][l] for l in range(self.n_layers)]),
            "lookback_ratio_mean": np.array([self.storage["lookback_ratio_mean"][l] for l in range(self.n_layers)]),
            "lookback_ratio_last": np.array([self.storage["lookback_ratio_last"][l] for l in range(self.n_layers)]),
            **out_stats,
        }
        return entry

    def process_dataset(
        self,
        corpus: List[Dict[str, Any]],
        output_file: str = "polygraph_data.pkl",
        max_new_tokens: int = 50,
        save_every: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Extracts mechanistic features across a batch corpus with periodic checkpointing.
        
        Args:
            corpus: List of input dictionaries with at least a 'text' and 'label' key.
            output_file: Path to serialize dataset checkpoints.
            max_new_tokens: Maximum tokens to generate per item.
            save_every: Frequency of disk serialization.
            
        Returns:
            List of processed feature entries.
        """
        dataset_features = []
        print(f"[FeatureExtractor] Starting extraction for {len(corpus)} prompts...")

        for idx, item in enumerate(corpus):
            prompt_text = item["text"]
            self._clean_memory()

            full_generation = self.generate_native(prompt_text, max_new_tokens=max_new_tokens)
            response_text = full_generation[len(prompt_text):].strip()

            prompt_tokens = self.model.to_tokens(prompt_text)
            self.current_prompt_len = prompt_tokens.shape[1]

            self.reset_storage()
            self.register_hooks()
            with torch.inference_mode():
                logits = self.model(full_generation)
            self.remove_hooks()

            out_stats = self._output_stats(logits, self.current_prompt_len)
            del logits

            entry = {
                "id": idx,
                "intended_label": item.get("label", "unspecified"),
                "condition": item.get("condition", "unspecified"),
                "outcome_label": None,
                "meta": item,
                "prompt": prompt_text,
                "response": response_text,
                "residual_means": np.array([self.storage["residual_means"][l] for l in range(self.n_layers)]),
                "residual_last": np.array([self.storage["residual_last"][l] for l in range(self.n_layers)]),
                "mlp_first": np.array([self.storage["mlp_first"][l] for l in range(self.n_layers)]),
                "mlp_last": np.array([self.storage["mlp_last"][l] for l in range(self.n_layers)]),
                "attn_entropy_mean": np.array([self.storage["attn_entropy_mean"][l] for l in range(self.n_layers)]),
                "attn_entropy_max": np.array([self.storage["attn_entropy_max"][l] for l in range(self.n_layers)]),
                "lookback_ratio_mean": np.array([self.storage["lookback_ratio_mean"][l] for l in range(self.n_layers)]),
                "lookback_ratio_last": np.array([self.storage["lookback_ratio_last"][l] for l in range(self.n_layers)]),
                **out_stats,
            }
            dataset_features.append(entry)
            print(f"[{idx+1}/{len(corpus)}] {item.get('label', '')} | {response_text[:60]}...")
            del entry, full_generation

            if (idx + 1) % save_every == 0:
                joblib.dump(dataset_features, output_file)
                print(f"  [Checkpoint saved at {idx+1} items]")

        joblib.dump(dataset_features, output_file)
        print(f"\n[FeatureExtractor] Extraction complete. Saved to '{output_file}'.")
        return dataset_features

    def process_and_append(
        self,
        new_items: List[Dict[str, Any]],
        existing_dataset: List[Dict[str, Any]],
        output_file: str,
        max_new_tokens: int = 60,
        temp_file: str = ".tmp_single_item.pkl",
    ) -> List[Dict[str, Any]]:
        """
        Incrementally processes and appends new items to an existing dataset list,
        saving to disk after each item to protect against VM interruptions.
        """
        next_id = max([e["id"] for e in existing_dataset], default=-1) + 1

        for i, item in enumerate(new_items):
            try:
                result = self.process_dataset(
                    [item],
                    output_file=temp_file,
                    max_new_tokens=max_new_tokens,
                    save_every=1,
                )
                entry = result[0]
                entry["id"] = next_id
                existing_dataset.append(entry)
                next_id += 1
                joblib.dump(existing_dataset, output_file)
                print(f"[{i+1}/{len(new_items)}] Progress saved - Total dataset size: {len(existing_dataset)}")
            except KeyboardInterrupt:
                print("\nInterrupted by user - all completed items up to now are safely saved.")
                raise
            except Exception as e:
                print(f"Error on item {i} (axis={item.get('axis', '?')}): {e} - skipping item.")
                continue

        print(f"\n[FeatureExtractor] Batch complete. Total entries: {len(existing_dataset)}")
        return existing_dataset

    def _clean_memory(self) -> None:
        """Invokes Python garbage collection and clears accelerator VRAM caches."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# Legacy Feature Extractors (Maintained for Backward Compatibility)
# ==============================================================================
class UnifiedFeatureExtractor:
    """Legacy feature extractor used in early explorations."""
    def __init__(self, model):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.reset_storage()

    def reset_storage(self):
        self.storage = {
            "residual_means": {},
            "residual_last": {},
            "attn_entropy": {},
            "lookback_ratio": {}
        }

    def register_hooks(self):
        self.hooks = []
        for layer in range(self.n_layers):
            res_hook = self.model.blocks[layer].hook_resid_post.add_hook(
                lambda val, hook, l=layer: self._hook_residual(val, hook, l)
            )
            attn_hook = self.model.blocks[layer].attn.hook_pattern.add_hook(
                lambda val, hook, l=layer: self._hook_attention(val, hook, l)
            )
            self.hooks.extend([res_hook, attn_hook])

    def remove_hooks(self):
        self.model.reset_hooks()

    def _hook_residual(self, value, hook, layer_idx):
        tensor = value.detach().cpu().to(torch.float32)
        self.storage["residual_means"][layer_idx] = torch.mean(tensor[0], dim=0).numpy()
        self.storage["residual_last"][layer_idx] = tensor[0, -1, :].numpy()
        return value

    def _hook_attention(self, value, hook, layer_idx):
        matrix = value.detach().cpu().to(torch.float32)[0]
        epsilon = 1e-9
        entropy_per_head = -torch.sum(matrix * torch.log(matrix + epsilon), dim=-1)
        mean_entropy = torch.mean(entropy_per_head, dim=-1).numpy()
        self.storage["attn_entropy"][layer_idx] = mean_entropy
        last_token_attn = matrix[:, -1, :]
        midpoint = max(1, last_token_attn.shape[-1] // 2)
        prompt_attn = torch.sum(last_token_attn[:, :midpoint], dim=-1)
        gen_attn = torch.sum(last_token_attn[:, midpoint:], dim=-1)
        ratio = (prompt_attn / (gen_attn + epsilon)).numpy()
        self.storage["lookback_ratio"][layer_idx] = ratio
        return value


class LocalFeatureExtractor(UnifiedFeatureExtractor):
    """Legacy local feature extractor used in early pilot runs."""
    pass
