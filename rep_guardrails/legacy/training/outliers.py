"""
Principled outlier detection using Chi-Square cutoff on squared standardized distances from centroid.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

from rep_guardrails.config import OUTLIER_CONFIDENCE


def detect_outliers_chisq(
    X_scaled: np.ndarray,
    confidence: float = OUTLIER_CONFIDENCE,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Identifies multidimensional outliers based on squared Euclidean distance from the standardized centroid.
    
    Under standard multivariate normality assumptions, the sum of squared standardized features
    approximates a Chi-Square distribution with degrees of freedom equal to the feature dimensionality (df).
    This establishes an objective cutoff independent of PCA axis alignment.
    
    Args:
        X_scaled: Standardized feature matrix (mean 0, variance 1 per column).
        confidence: Critical value threshold (e.g. 0.995 for conservative filtering).
        
    Returns:
        Tuple of (is_outlier_mask, cutoff_threshold, squared_distances_array).
    """
    sq_dist = np.sum(X_scaled**2, axis=1)
    df = X_scaled.shape[1]
    cutoff = float(chi2.ppf(confidence, df))
    is_outlier = sq_dist > cutoff
    return is_outlier, cutoff, sq_dist


def filter_outliers(
    X: np.ndarray,
    dataset: List[Dict[str, Any]],
    confidence: float = OUTLIER_CONFIDENCE,
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """
    Standardizes the feature matrix, detects extreme anomalies via Chi-Square cutoff,
    and returns the cleaned feature matrix and corresponding dataset entries.
    
    Args:
        X: Raw flattened feature matrix of shape (N, D).
        dataset: Parallel list of dataset dictionaries.
        confidence: Chi-Square confidence cutoff.
        
    Returns:
        Tuple of (X_clean, dataset_clean, is_outlier_mask).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    is_outlier, cutoff, sq_dist = detect_outliers_chisq(X_scaled, confidence=confidence)

    n_flagged = int(is_outlier.sum())
    print(f"[OutlierFilter] Confidence: {confidence*100:.1f}% | Cutoff distance²: {cutoff:.1f}")
    print(f"[OutlierFilter] Flagged {n_flagged} outlier(s) out of {len(dataset)} entries.")

    if n_flagged > 0:
        for idx in np.where(is_outlier)[0]:
            meta = dataset[idx].get("meta", {})
            domain = meta.get("domain", "unknown")
            resp = dataset[idx].get("response", "")[:60]
            print(f"  -> Flagged ID {dataset[idx].get('id', idx)} | Domain: {domain} | Dist²: {sq_dist[idx]:.1f} | '{resp}...'")

    clean_mask = ~is_outlier
    X_clean = X[clean_mask]
    dataset_clean = [dataset[i] for i in range(len(dataset)) if clean_mask[i]]

    return X_clean, dataset_clean, is_outlier
