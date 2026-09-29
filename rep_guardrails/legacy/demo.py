#!/usr/bin/env python3
"""
Quick demo script for Representation Engineering Guardrails.

Supports:
  1. Instant Offline Verification: Evaluates pre-computed activations from
     'political_probe_dataset.pkl' against the trained classifier probe.
  2. Live Model Guardrail: Loads Llama-3-8B-Instruct and runs live hooked
     generation and verification on custom queries.
"""

import argparse
import sys
import os
import joblib
import numpy as np

from rep_guardrails.config import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_KEY,
    SEVERITY_THRESHOLD,
)
from rep_guardrails.extraction.feature_utils import flatten_entry
from rep_guardrails.agents.guardrail_agent import GuardrailAgent


def run_offline_verification(classifier_path: str, dataset_path: str) -> None:
    """Demonstrates probe classification on real sampled activation tensors from disk."""
    print("\n" + "=" * 65)
    print("   RUNNING INSTANT OFFLINE VERIFICATION DEMO")
    print("=" * 65)

    if not os.path.exists(classifier_path):
        print(f"Error: Classifier not found at '{classifier_path}'.")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        return

    print(f"Loading trained classifier probe: '{classifier_path}'...")
    clf = joblib.load(classifier_path)

    print(f"Loading dataset: '{dataset_path}'...")
    data = joblib.load(dataset_path)
    print(f"Loaded {len(data)} total samples.")

    # Select representative samples: one political, one control
    pol_samples = [e for e in data if e.get("meta", {}).get("domain") == "political"]
    unrelated_samples = [e for e in data if e.get("meta", {}).get("domain") == "unrelated"]

    test_entries = []
    if pol_samples:
        test_entries.append(("Target Domain (Political)", pol_samples[0]))
    if unrelated_samples:
        test_entries.append(("Control Domain (Non-political)", unrelated_samples[0]))

    print("\n" + "-" * 65)
    for category, entry in test_entries:
        feat_vec = flatten_entry(entry).reshape(1, -1)
        pred = int(clf.predict(feat_vec)[0])
        prob = float(clf.predict_proba(feat_vec)[0][1])
        relevance = entry.get("response_political_relevance", 0.0)

        prompt_preview = entry.get("meta", {}).get("raw_question", entry.get("prompt", ""))[:55]
        response_preview = entry.get("response", "")[:60]

        status = "\033[91m[FLAGGED / INTERCEPTED]\033[0m" if pred == 1 else "\033[92m[CLEARED / ALLOWED]\033[0m"

        print(f"Category:     {category}")
        print(f"Question:     '{prompt_preview}...'")
        print(f"LLM Response: '{response_preview}...'")
        print(f"NLI Score:    {relevance:.4f}")
        print(f"Probe Result: {status} (P(target)={prob:.2%}, Decision={pred})")
        print("-" * 65)

    print("\nOffline verification completed successfully.")
    print("To test interactive live inference with Llama-3, run:")
    print("  python demo.py --live\n")


def run_live_guardrail(model_key: str, classifier_path: str, question: str = None) -> None:
    """Runs the live guardrail agent with native generation and hooked forward pass."""
    print("\n" + "=" * 65)
    print("   LAUNCHING LIVE GUARDRAIL AGENT (LLAMA-3-8B)")
    print("=" * 65)

    agent = GuardrailAgent(
        classifier_path=classifier_path,
        model_key=model_key,
    )

    if question:
        agent.evaluate(question)
    else:
        agent.interactive_session()


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation Guardrails Demonstration")
    parser.add_argument("--live", action="store_true", help="Run live Llama-3 model generation and verification")
    parser.add_argument("-q", "--question", type=str, help="Specific question for live guardrail")
    parser.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER_PATH, help="Path to classifier probe")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY, help="LLM model key")

    args = parser.parse_args()

    if args.live or args.question:
        run_live_guardrail(
            model_key=args.model,
            classifier_path=args.classifier,
            question=args.question,
        )
    else:
        run_offline_verification(
            classifier_path=args.classifier,
            dataset_path=args.dataset,
        )


if __name__ == "__main__":
    main()
