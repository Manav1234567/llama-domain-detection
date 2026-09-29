"""
Training, benchmarking, and export of linear probes and regression models
on internal activation representations.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from rep_guardrails.config import (
    SEVERITY_THRESHOLD,
    N_PCA_COMPONENTS,
    DEFAULT_CLASSIFIER_PATH,
)
from rep_guardrails.extraction.feature_utils import flatten_entry, extract_feature_matrix
from rep_guardrails.training.outliers import filter_outliers
from rep_guardrails.training.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)


class ProbeTrainer:
    """
    Orchestrates data preparation, cross-model benchmarking, and production training
    of linear probes for latent domain detection and severity grading.
    """

    def __init__(
        self,
        severity_threshold: float = SEVERITY_THRESHOLD,
        n_pca_components: int = N_PCA_COMPONENTS,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.n_pca_components = n_pca_components

    def prepare_data(
        self,
        dataset: List[Dict[str, Any]],
        filter_extreme_outliers: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts feature matrix X and ground-truth targets from dataset entries.
        
        Args:
            dataset: List of dataset dictionaries containing extracted features and scores.
            filter_extreme_outliers: If True, eliminates Chi-Square distribution outliers.
            
        Returns:
            Tuple of (X, y_binary, relevance_scores, cleaned_dataset).
        """
        X = extract_feature_matrix(dataset)

        if filter_extreme_outliers:
            X, dataset, _ = filter_outliers(X, dataset)

        relevance = np.array([
            e.get("response_political_relevance", e.get("meta", {}).get("severity", 0.0))
            for e in dataset
        ], dtype=np.float32)

        # Ground truth outcome label: 1 if response exceeded severity threshold, 0 otherwise
        y_binary = (relevance > self.severity_threshold).astype(int)

        print(f"[ProbeTrainer] Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features.")
        print(f"[ProbeTrainer] Positive class ratio: {y_binary.mean()*100:.1f}% (threshold > {self.severity_threshold})")
        return X, y_binary, relevance, dataset

    def benchmark_classifiers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_components: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
        """
        Evaluates multiple classification probe heads across dimensionality-reduced activations.
        
        Models compared:
          - Logistic Regression (Linear baseline & production candidate)
          - SVM (Linear)
          - SVM (RBF kernel)
          - Random Forest
          - Decision Tree
          
        Returns:
            Tuple of (summary_results_dataframe, dictionary_of_fitted_pipelines).
        """
        pca_k = n_components or min(self.n_pca_components, X_train.shape[0] - 1)
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "SVM (Linear)": SVC(
                kernel="linear", class_weight="balanced", probability=True, random_state=random_state
            ),
            "SVM (RBF)": SVC(
                kernel="rbf", class_weight="balanced", probability=True, random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=random_state
            ),
        }

        results = []
        fitted_pipes: Dict[str, Pipeline] = {}

        print(f"[ProbeTrainer] Benchmarking {len(models)} classification probes (PCA components={pca_k})...")
        for name, clf in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_k, random_state=random_state)),
                ("clf", clf),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = compute_classification_metrics(y_test, y_pred)
            metrics["model"] = name
            results.append(metrics)
            fitted_pipes[name] = pipe

        df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1"]].sort_values(
            "f1", ascending=False
        )
        return df, fitted_pipes

    def benchmark_regressors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_components: Optional[int] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
        """
        Evaluates continuous regression models to grade severity of domain presence.
        
        Models compared:
          - Ridge Regression
          - Support Vector Regression (SVR RBF)
          - Random Forest Regressor
          - Decision Tree Regressor
          
        Returns:
            Tuple of (summary_results_dataframe, dictionary_of_fitted_pipelines).
        """
        pca_k = n_components or min(self.n_pca_components, X_train.shape[0] - 1)
        models = {
            "Ridge": Ridge(alpha=5.0, random_state=random_state),
            "SVR (RBF)": SVR(kernel="rbf", C=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=random_state
            ),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        }

        results = []
        fitted_pipes: Dict[str, Pipeline] = {}

        print(f"[ProbeTrainer] Benchmarking {len(models)} regression models (PCA components={pca_k})...")
        for name, reg in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_k, random_state=random_state)),
                ("reg", reg),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = compute_regression_metrics(y_test, y_pred)
            metrics["model"] = name
            results.append(metrics)
            fitted_pipes[name] = pipe

        df = pd.DataFrame(results)[["model", "r2", "mae", "rmse"]].sort_values("r2", ascending=False)
        return df, fitted_pipes

    def train_final_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: Optional[int] = None,
        output_path: str = DEFAULT_CLASSIFIER_PATH,
        random_state: int = 42,
    ) -> Pipeline:
        """
        Fits the production guardrail linear probe on the complete dataset and exports to disk.
        
        Pipeline stages:
          1. StandardScaler (Feature standardization)
          2. PCA (Dimensionality compression to n_components)
          3. LogisticRegression (Calibrated linear boundary with balanced class weighting)
          
        Args:
            X: Complete feature matrix.
            y: Binary target labels (1 = target domain, 0 = benign).
            n_components: Number of PCA components.
            output_path: File destination to serialize the fitted pipeline.
            random_state: Random state for reproducibility.
            
        Returns:
            Fitted scikit-learn Pipeline.
        """
        pca_k = n_components or min(self.n_pca_components, X.shape[0] - 1)
        print(f"[ProbeTrainer] Fitting production probe on {X.shape[0]} samples with {pca_k} PCA components...")

        final_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_k, random_state=random_state)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])

        final_pipe.fit(X, y)

        joblib.dump(final_pipe, output_path)
        print(f"[ProbeTrainer] Production classifier saved successfully to '{output_path}'.")
        return final_pipe

    @staticmethod
    def load_classifier(filepath: str = DEFAULT_CLASSIFIER_PATH) -> Pipeline:
        """Loads a pre-trained guardrail pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Classifier file not found at '{filepath}'.")
        return joblib.load(filepath)
