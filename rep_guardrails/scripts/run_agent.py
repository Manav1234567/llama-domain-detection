#!/usr/bin/env python3
"""
CLI entry point to launch and orchestrate autonomous agents in the
Representation Engineering Guardrail framework.

Usage examples:
    # Run interactive guardrail agent kill switch
    python run_agent.py guardrail --interactive

    # Evaluate a single query through the guardrail
    python run_agent.py guardrail --question "What are the arguments for and against universal healthcare?"

    # Train and evaluate linear probes on activation data
    python run_agent.py train --dataset political_probe_dataset.pkl --benchmark

    # Run latent PCA and feature importance diagnostics
    python run_agent.py analyze --dataset political_probe_dataset.pkl

    # Collect activation representations for new questions
    python run_agent.py collect --target-per-axis 20 --target-unrelated 20

    # Score responses with zero-shot DeBERTa judge
    python run_agent.py judge --dataset political_probe_dataset.pkl
"""

import argparse
import sys
import os

from rep_guardrails.config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_JUDGE_MODEL,
    SEVERITY_THRESHOLD,
)
from rep_guardrails.agents import (
    GuardrailAgent,
    ProbeTrainingAgent,
    AnalysisAgent,
    DataCollectionAgent,
    JudgeAgent,
)


def handle_guardrail(args: argparse.Namespace) -> None:
    """Handles the inference guardrail agent subcommand."""
    print("[CLI] Initializing GuardrailAgent...")
    agent = GuardrailAgent(
        classifier_path=args.classifier,
        model_key=args.model,
        threshold=args.threshold,
    )

    if args.interactive:
        agent.interactive_session(max_new_tokens=args.max_tokens)
    elif args.question:
        result = agent.evaluate(args.question, max_new_tokens=args.max_tokens)
        print("\n" + "=" * 50)
        print(f"Question:    {result.prompt}")
        print(f"Status:      {'INTERCEPTED (Political)' if result.is_flagged else 'PASSED (Benign)'}")
        print(f"Probability: {result.probability:.4f} (Threshold: {result.threshold})")
        print(f"Output:      {result.final_response}")
        print(f"Latency:     Gen={result.generation_time_ms:.1f}ms | Verify={result.verification_time_ms:.1f}ms")
        print("=" * 50)
    else:
        print("Error: Specify either --question \"...\" or --interactive to run the guardrail agent.")
        sys.exit(1)


def handle_train(args: argparse.Namespace) -> None:
    """Handles the probe training agent subcommand."""
    print(f"[CLI] Launching ProbeTrainingAgent with dataset '{args.dataset}'...")
    agent = ProbeTrainingAgent(
        dataset_path=args.dataset,
        output_classifier_path=args.output,
        severity_threshold=args.threshold,
    )
    pipe, benchmark_df, group_imp = agent.train_and_evaluate(
        run_benchmark=args.benchmark,
        run_importance=args.importance,
        filter_outliers=not args.no_outlier_filter,
    )
    print(f"\n[CLI] Probe training completed. Model saved to '{args.output}'.")


def handle_analyze(args: argparse.Namespace) -> None:
    """Handles the latent analysis agent subcommand."""
    print(f"[CLI] Launching AnalysisAgent on dataset '{args.dataset}'...")
    agent = AnalysisAgent(dataset_path=args.dataset)
    summary = agent.analyze(
        save_pca_plot=args.pca_out,
        save_loadings_plot=args.loadings_out,
        filter_extreme_outliers=not args.no_outlier_filter,
        show_plots=args.show,
    )
    print("\n[CLI] Analysis complete.")
    print(f"Total samples analyzed: {summary['num_samples']}")
    print(f"PCA Variance Explained: PC1={summary['explained_variance_ratio'][0]*100:.2f}%, PC2={summary['explained_variance_ratio'][1]*100:.2f}%")


def handle_collect(args: argparse.Namespace) -> None:
    """Handles the data collection agent subcommand."""
    print(f"[CLI] Launching DataCollectionAgent (target_per_axis={args.target_per_axis}, target_unrelated={args.target_unrelated})...")
    agent = DataCollectionAgent(
        model_key=args.model,
        output_file=args.output,
    )
    dataset = agent.collect(
        target_per_axis=args.target_per_axis,
        target_unrelated=args.target_unrelated,
        max_new_tokens=args.max_tokens,
    )
    print(f"\n[CLI] Collection completed. Total dataset entries: {len(dataset)}")


def handle_judge(args: argparse.Namespace) -> None:
    """Handles the ground-truth scoring agent subcommand."""
    print(f"[CLI] Launching JudgeAgent with model '{args.model_name}' on '{args.dataset}'...")
    agent = JudgeAgent(
        dataset_path=args.dataset,
        model_name=args.model_name,
    )
    dataset = agent.score()
    print(f"\n[CLI] Scoring completed. Annotated {len(dataset)} responses.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI tool to run Representation Engineering Guardrail agents."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Agent action to execute")

    # 1. Guardrail Agent Subcommand
    p_guardrail = subparsers.add_parser("guardrail", help="Run inference kill-switch guardrail agent")
    p_guardrail.add_argument("-q", "--question", type=str, help="Single question to evaluate")
    p_guardrail.add_argument("-i", "--interactive", action="store_true", help="Launch interactive CLI chat session")
    p_guardrail.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY, help="LLM model key")
    p_guardrail.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER_PATH, help="Path to trained probe pipeline")
    p_guardrail.add_argument("--threshold", type=float, default=SEVERITY_THRESHOLD, help="Intervention decision threshold")
    p_guardrail.add_argument("--max-tokens", type=int, default=60, help="Max generation tokens")

    # 2. Train Agent Subcommand
    p_train = subparsers.add_parser("train", help="Run probe training agent")
    p_train.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to activation dataset")
    p_train.add_argument("--output", type=str, default=DEFAULT_CLASSIFIER_PATH, help="Path to save trained classifier")
    p_train.add_argument("--threshold", type=float, default=SEVERITY_THRESHOLD, help="Binary classification relevance threshold")
    p_train.add_argument("--benchmark", action="store_true", default=True, help="Run multi-model classifier benchmark")
    p_train.add_argument("--no-benchmark", dest="benchmark", action="store_false", help="Skip multi-model benchmark")
    p_train.add_argument("--importance", action="store_true", default=True, help="Compute grouped permutation feature importance")
    p_train.add_argument("--no-importance", dest="importance", action="store_false", help="Skip permutation importance")
    p_train.add_argument("--no-outlier-filter", action="store_true", help="Skip Chi-Square outlier detection")

    # 3. Analyze Agent Subcommand
    p_analyze = subparsers.add_parser("analyze", help="Run latent analysis & visualization agent")
    p_analyze.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset")
    p_analyze.add_argument("--pca-out", type=str, default="latent_pca_projection.png", help="Path to save PCA scatter plot")
    p_analyze.add_argument("--loadings-out", type=str, default="feature_loadings_importance.png", help="Path to save loadings plot")
    p_analyze.add_argument("--no-outlier-filter", action="store_true", help="Skip outlier filtering")
    p_analyze.add_argument("--show", action="store_true", help="Display interactive plot GUI window")

    # 4. Collect Agent Subcommand
    p_collect = subparsers.add_parser("collect", help="Run dataset collection agent")
    p_collect.add_argument("--target-per-axis", type=int, default=20, help="Target questions per policy axis")
    p_collect.add_argument("--target-unrelated", type=int, default=20, help="Target control non-political prompts")
    p_collect.add_argument("--model", type=str, default=DEFAULT_MODEL_KEY, help="Model key for generation")
    p_collect.add_argument("--output", type=str, default=DEFAULT_DATASET_PATH, help="Output dataset path")
    p_collect.add_argument("--max-tokens", type=int, default=60, help="Max generation tokens")

    # 5. Judge Agent Subcommand
    p_judge = subparsers.add_parser("judge", help="Run zero-shot DeBERTa judge agent")
    p_judge.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to dataset to annotate")
    p_judge.add_argument("--model-name", type=str, default=DEFAULT_JUDGE_MODEL, help="DeBERTa zero-shot model name")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "guardrail": handle_guardrail,
        "train": handle_train,
        "analyze": handle_analyze,
        "collect": handle_collect,
        "judge": handle_judge,
    }

    handlers[args.subcommand](args)


if __name__ == "__main__":
    main()
