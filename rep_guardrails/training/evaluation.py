"""
Model evaluation, grouped permutation importance, and performance metrics
for classification and regression probes.
"""

from typing import Dict, Tuple, Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes standard binary classification evaluation metrics.
    
    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard continuous regression evaluation metrics.
    
    Returns:
        Dictionary containing MAE, RMSE, and R2 score.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def grouped_permutation_importance(
    pipe: Any,
    X: np.ndarray,
    y: np.ndarray,
    group_array: np.ndarray,
    scoring_func: Callable[[np.ndarray, np.ndarray], float],
    higher_is_better: bool = True,
    n_repeats: int = 8,
    seed: int = 42,
) -> Tuple[pd.Series, float]:
    """
    Computes grouped permutation feature importance for an arbitrary scikit-learn pipeline.
    
    Shuffles all columns belonging to a group simultaneously to preserve intra-group correlations
    while measuring the aggregate contribution of the conceptual feature group or transformer layer.
    
    Args:
        pipe: Fitted scikit-learn Pipeline or estimator.
        X: Feature matrix of shape (N, D).
        y: Target ground truth vector.
        group_array: Parallel 1D array of length D labeling group membership for each feature.
        scoring_func: Callable taking (y_true, y_pred) and returning a scalar score.
        higher_is_better: True if larger score is better (e.g. F1, R2); False for error metrics.
        n_repeats: Number of permutation iterations per group.
        seed: Random seed for shuffling.
        
    Returns:
        Tuple of (importance_series_sorted_descending, baseline_score).
    """
    rng = np.random.default_rng(seed)
    baseline_pred = pipe.predict(X)
    baseline = scoring_func(y, baseline_pred)
    importances: Dict[Any, float] = {}

    unique_groups = np.unique(group_array)
    for g in unique_groups:
        mask = (group_array == g)
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(X.shape[0])
            # Permute rows across the active columns in group g
            X_perm[:, mask] = X_perm[perm_idx][:, mask]
            score = scoring_func(y, pipe.predict(X_perm))
            drop = (baseline - score) if higher_is_better else (score - baseline)
            drops.append(drop)
        importances[g] = float(np.mean(drops))

    series = pd.Series(importances).sort_values(ascending=False)
    return series, float(baseline)
