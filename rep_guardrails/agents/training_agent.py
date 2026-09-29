"""
Probe Training Agent for executing outlier filtering, cross-model benchmarking,
permutation feature importance, and production linear probe training.
"""

from typing import Optional, Dict, Any, Tuple
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from rep_guardrails.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_CLASSIFIER_PATH,
    SEVERITY_THRESHOLD,
    N_PCA_COMPONENTS,
)
from rep_guardrails.data.corpus import load_existing_dataset
from rep_guardrails.extraction.feature_utils import build_index_maps
from rep_guardrails.training.probe_trainer import ProbeTrainer
from rep_guardrails.training.evaluation import grouped_permutation_importance
from rep_guardrails.agents.base_agent import BaseAgent


class ProbeTrainingAgent(BaseAgent):
    """
    Autonomous agent managing statistical probe training, hyperparameter configuration,
    architectural benchmarking, and production model serialization.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        output_classifier_path: str = DEFAULT_CLASSIFIER_PATH,
        severity_threshold: float = SEVERITY_THRESHOLD,
        n_pca_components: int = N_PCA_COMPONENTS,
    ) -> None:
        super().__init__(name="ProbeTrainingAgent")
        self.dataset_path = dataset_path
        self.output_classifier_path = output_classifier_path
        self.trainer = ProbeTrainer(
            severity_threshold=severity_threshold,
            n_pca_components=n_pca_components,
        )

    def train_and_evaluate(
        self,
        run_benchmark: bool = True,
        run_importance: bool = True,
        filter_outliers: bool = True,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Tuple[Pipeline, Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Executes end-to-end training and evaluation workflow.
        
        Args:
            run_benchmark: If True, evaluates multi-model classification comparison.
            run_importance: If True, computes grouped permutation importance by feature group and layer.
            filter_outliers: If True, detects and removes Chi-Square distribution outliers.
            test_size: Holdout evaluation proportion.
            seed: Random seed.
            
        Returns:
            Tuple of (production_pipeline, benchmark_dataframe, group_importance_series).
        """
        self.log(f"Loading dataset from '{self.dataset_path}'...")
        dataset = load_existing_dataset(self.dataset_path)
        if not dataset:
            raise FileNotFoundError(f"Dataset at '{self.dataset_path}' is empty or not found.")

        # 1. Feature extraction and outlier filtering
        self.log("Preparing features and computing outcome ground-truth labels...")
        X, y, relevance, clean_dataset = self.trainer.prepare_data(
            dataset, filter_extreme_outliers=filter_outliers
        )

        benchmark_df: Optional[pd.DataFrame] = None
        group_imp: Optional[pd.Series] = None

        # 2. Benchmark classifiers if requested
        if run_benchmark:
            self.log(f"Splitting train/test sets (test_size={test_size}, stratified)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=seed
            )
            self.log(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

            benchmark_df, _ = self.trainer.benchmark_classifiers(
                X_train, y_train, X_test, y_test, random_state=seed
            )
            self.log(f"\nClassifier Benchmark Results:\n{benchmark_df.to_string(index=False)}")

        # 3. Permutation feature importance analysis
        if run_importance and len(clean_dataset) > 0:
            self.log("Calculating grouped permutation feature and layer importance...")
            group_labels, layer_labels = build_index_maps(clean_dataset[0])
            f1_scorer = lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)

            # Fit standard reference pipeline for evaluation
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=seed
            )
            ref_pipe = self.trainer.train_final_classifier(
                X_tr, y_tr, output_path=".tmp_ref_pipe.pkl", random_state=seed
            )

            group_imp, baseline = grouped_permutation_importance(
                ref_pipe, X_te, y_te, group_labels, f1_scorer, seed=seed
            )
            layer_imp, _ = grouped_permutation_importance(
                ref_pipe, X_te, y_te, layer_labels, f1_scorer, seed=seed
            )
            layer_imp = layer_imp[layer_imp.index >= 0].sort_index()

            self.log(f"Baseline F1 score: {baseline:.4f}")
            self.log(f"\nTop Feature Groups by F1 Drop:\n{group_imp.to_string()}")
            self.log(f"\nTop Transformer Layers by F1 Drop:\n{layer_imp.sort_values(ascending=False).head(5).to_string()}")

            if os.path.exists(".tmp_ref_pipe.pkl"):
                os.remove(".tmp_ref_pipe.pkl")

        # 4. Train and export final production pipeline on all available samples
        self.log(f"Training production probe on full {X.shape[0]} samples...")
        final_pipeline = self.trainer.train_final_classifier(
            X, y, output_path=self.output_classifier_path, random_state=seed
        )

        self.log(f"Probe training complete. Artifact saved to '{self.output_classifier_path}'.")
        return final_pipeline, benchmark_df, group_imp

    def run(self, run_benchmark: bool = True) -> Pipeline:
        """Agent entrypoint matching BaseAgent interface."""
        pipe, _, _ = self.train_and_evaluate(run_benchmark=run_benchmark)
        return pipe
