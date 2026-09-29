"""
Inference-Time Guardrail Agent (The 'Streaming Safeguard' Kill-Switch).

Executes decoupled 'Fast Generation, Hooked Verification' to detect and intercept
unauthorized domain discourse before output is delivered to the user.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time
import html
import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from rep_guardrails.config import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_MODEL_KEY,
    DEFAULT_INTERCEPTION_MESSAGE,
    SEVERITY_THRESHOLD,
    get_device,
    get_default_dtype,
)
from rep_guardrails.models.loader import load_model_bundle
from rep_guardrails.extraction.feature_extractor import PolygraphFeatureExtractor
from rep_guardrails.extraction.feature_utils import flatten_entry
from rep_guardrails.agents.base_agent import BaseAgent


@dataclass
class GuardrailResult:
    """Structured telemetry output returned by the GuardrailAgent for each query."""
    prompt: str
    raw_response: str
    final_response: str
    is_flagged: bool
    probability: float
    threshold: float
    generation_time_ms: float
    verification_time_ms: float
    total_latency_ms: float

    @property
    def was_intercepted(self) -> bool:
        """Returns True if the response was blocked and replaced by the guardrail."""
        return self.is_flagged


class GuardrailAgent(BaseAgent):
    """
    Inference guardrail agent monitoring LLM activations in real-time.
    
    Upon user query:
      1. Generates response using fast native generation kernels.
      2. Runs a single non-autoregressive hooked forward pass.
      3. Evaluates intermediate residual stream, attention entropy, and lookback ratios via a linear probe.
      4. Halts and overrides unauthorized conceptual output before it is returned.
    """

    def __init__(
        self,
        classifier_path: str = DEFAULT_CLASSIFIER_PATH,
        classifier_pipe: Optional[Pipeline] = None,
        model_key: str = DEFAULT_MODEL_KEY,
        model_bundle: Optional[Tuple[HookedTransformer, AutoModelForCausalLM, AutoTokenizer]] = None,
        interception_message: str = DEFAULT_INTERCEPTION_MESSAGE,
        threshold: float = SEVERITY_THRESHOLD,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(name="GuardrailAgent")
        self.classifier_path = classifier_path
        self.model_key = model_key
        self.interception_message = interception_message
        self.threshold = threshold
        self.device = device or get_device()
        self.dtype = dtype or get_default_dtype(self.device)

        # 1. Initialize classifier pipeline
        if classifier_pipe is not None:
            self.classifier = classifier_pipe
        else:
            self.log(f"Loading classifier probe from '{classifier_path}'...")
            self.classifier = joblib.load(classifier_path)

        # 2. Initialize LLM models and extractor
        if model_bundle is not None:
            self.hooked_model, self.hf_model, self.tokenizer = model_bundle
        else:
            self.hooked_model, self.hf_model, self.tokenizer = None, None, None

        self.extractor: Optional[PolygraphFeatureExtractor] = None
        if self.hooked_model is not None:
            self._init_extractor()

    def _init_extractor(self) -> None:
        self.extractor = PolygraphFeatureExtractor(
            model=self.hooked_model,
            hf_gen_model=self.hf_model,
            hf_tokenizer=self.tokenizer,
            device=self.device,
        )

    def ensure_models_loaded(self) -> None:
        """Loads LLM weights if not already loaded into memory."""
        if self.extractor is None:
            self.log(f"Loading model bundle for '{self.model_key}' on {self.device}...")
            self.hooked_model, self.hf_model, self.tokenizer = load_model_bundle(
                self.model_key, device=self.device, dtype=self.dtype
            )
            self._init_extractor()

    def evaluate(self, question_text: str, max_new_tokens: int = 60) -> GuardrailResult:
        """
        Processes a user question, runs native generation, extracts latent activations,
        and enforces the guardrail kill switch.
        
        Args:
            question_text: The incoming user query.
            max_new_tokens: Maximum tokens to generate.
            
        Returns:
            GuardrailResult with decisions, raw text, and latency metrics.
        """
        self.ensure_models_loaded()
        assert self.extractor is not None

        # 1. Format user prompt with model's native chat template
        messages = [{"role": "user", "content": question_text}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Fast native C++ generation
        t_gen_start = time.perf_counter()
        full_generation = self.extractor.generate_native(prompt_text, max_new_tokens=max_new_tokens)
        gen_time_ms = (time.perf_counter() - t_gen_start) * 1000.0

        raw_response = full_generation[len(prompt_text):].strip()

        # 3. Hooked forward pass to extract mechanistic feature tensors
        t_verify_start = time.perf_counter()
        prompt_tokens = self.extractor.model.to_tokens(prompt_text)
        self.extractor.current_prompt_len = prompt_tokens.shape[1]

        self.extractor.reset_storage()
        self.extractor.register_hooks()
        with torch.inference_mode():
            logits = self.extractor.model(full_generation)
        self.extractor.remove_hooks()

        # 4. Construct unified feature representation vector
        entry: Dict[str, Any] = {
            k: np.array([self.extractor.storage[k][l] for l in range(self.extractor.n_layers)])
            for k in [
                "residual_means", "residual_last", "mlp_first", "mlp_last",
                "attn_entropy_mean", "attn_entropy_max",
                "lookback_ratio_mean", "lookback_ratio_last",
            ]
        }
        entry.update(self.extractor._output_stats(logits, self.extractor.current_prompt_len))
        del logits

        feat_vec = flatten_entry(entry).reshape(1, -1)

        # 5. Evaluate linear probe
        pred = int(self.classifier.predict(feat_vec)[0])
        prob = float(self.classifier.predict_proba(feat_vec)[0][1])
        verify_time_ms = (time.perf_counter() - t_verify_start) * 1000.0

        is_flagged = bool(pred == 1)
        final_response = self.interception_message if is_flagged else raw_response

        total_ms = gen_time_ms + verify_time_ms

        result = GuardrailResult(
            prompt=question_text,
            raw_response=raw_response,
            final_response=final_response,
            is_flagged=is_flagged,
            probability=prob,
            threshold=self.threshold,
            generation_time_ms=gen_time_ms,
            verification_time_ms=verify_time_ms,
            total_latency_ms=total_ms,
        )

        status_str = "INTERCEPTED" if is_flagged else "PASSED"
        self.log(
            f"Query: '{question_text[:40]}...' -> [{status_str}] "
            f"P(target)={prob:.4f} | Gen: {gen_time_ms:.1f}ms | Verify: {verify_time_ms:.1f}ms"
        )
        return result

    def run(self, question: str, max_new_tokens: int = 60) -> GuardrailResult:
        """Agent entrypoint matching BaseAgent interface."""
        return self.evaluate(question, max_new_tokens=max_new_tokens)

    def interactive_session(self, max_new_tokens: int = 60) -> None:
        """Runs a continuous interactive CLI chat loop with color-coded guardrail output."""
        self.ensure_models_loaded()
        print("\n" + "=" * 65)
        print("   REPRESENTATION GUARDRAIL AGENT - INTERACTIVE SESSION")
        print("   Type 'exit', 'quit', or press Ctrl+C to terminate.")
        print("=" * 65 + "\n")

        while True:
            try:
                query = input("\n[User Question] > ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive session.")
                    break

                res = self.evaluate(query, max_new_tokens=max_new_tokens)

                print("\n" + "-" * 50)
                if res.is_flagged:
                    print(f"\033[91m[GUARDRAIL TRIGGERED - {res.probability:.2%} Confidence]\033[0m")
                    print(f"\033[91mRaw Model Output:\033[0m {res.raw_response}")
                    print(f"\033[93mEnforced Override:\033[0m {res.final_response}")
                else:
                    print(f"\033[92m[CLEARED - {1.0 - res.probability:.2%} Non-Political Confidence]\033[0m")
                    print(f"\033[92mResponse:\033[0m {res.final_response}")

                print(f"[Latency] Generation: {res.generation_time_ms:.1f}ms | Verification: {res.verification_time_ms:.1f}ms")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\nSession aborted by user.")
                break

    @staticmethod
    def render_html(res: GuardrailResult) -> str:
        """
        Renders a styled HTML display card representing the inspection result.
        """
        flag_color = "#e74c3c" if res.is_flagged else "#2ecc71"
        flag_text = "⚠ Political (Intercepted)" if res.is_flagged else "✓ Non-political (Allowed)"
        display_prob = res.probability if res.is_flagged else (1.0 - res.probability)

        return f"""
        <div style="background:#1e1e1e; border:1px solid #3c3c3c; border-radius:12px; padding:18px; margin:12px 0; font-family:'Segoe UI', Arial, sans-serif; color:#d4d4d4; box-shadow:0 2px 8px rgba(0,0,0,0.35);">
            <div style="font-size:13px; color:#9cdcfe; margin-bottom:8px;"><b>User Prompt</b></div>
            <div style="font-size:15px; color:#ffffff; margin-bottom:14px; line-height:1.5;">{html.escape(res.prompt)}</div>
            <div style="font-size:13px; color:#ce9178; margin-bottom:6px;"><b>Delivered Output</b></div>
            <div style="background:#2a2a2a; padding:12px; border-radius:6px; white-space:pre-wrap; line-height:1.5; font-size:14px;">{html.escape(res.final_response)}</div>
            <div style="margin-top:14px; display:flex; align-items:center;">
                <span style="background:{flag_color}; color:white; padding:4px 12px; border-radius:12px; font-size:12px; font-weight:bold;">{flag_text}</span>
                <span style="margin-left:12px; color:#aaa; font-size:13px;">Confidence: <b>{display_prob:.2%}</b></span>
                <span style="margin-left:auto; color:#888; font-size:12px;">Lat: {res.total_latency_ms:.1f}ms</span>
            </div>
        </div>
        """
