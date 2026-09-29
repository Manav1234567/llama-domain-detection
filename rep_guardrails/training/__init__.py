"""
Model training, evaluation, and outlier detection subpackage.
"""

from rep_guardrails.training.outliers import detect_outliers_chisq, filter_outliers
from rep_guardrails.training.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    grouped_permutation_importance,
)
from rep_guardrails.training.probe_trainer import ProbeTrainer

__all__ = [
    "detect_outliers_chisq",
    "filter_outliers",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "grouped_permutation_importance",
    "ProbeTrainer",
]
