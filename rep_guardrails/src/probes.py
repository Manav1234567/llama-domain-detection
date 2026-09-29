from typing import Dict, List, Tuple, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from src.config import (
    FEATURE_KEYS,
    SCALAR_KEYS,
    OUTLIER_CONFIDENCE,
    SEVERITY_THRESHOLD,
    N_PCA_COMPONENTS,
    DEFAULT_CLASSIFIER_PATH,
)


def flatten_entry(
    entry: Dict[str, Any],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> np.ndarray:
    f_keys = feature_keys or FEATURE_KEYS
    s_keys = scalar_keys or SCALAR_KEYS

    parts = [entry[k].flatten().astype(np.float32) for k in f_keys]
    scalar_vals = [entry[k] for k in s_keys]
    parts.append(np.array(scalar_vals, dtype=np.float32))
    return np.concatenate(parts)


def get_feature_index_map(
    entry: Dict[str, Any],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> List[Tuple[str, int]]:
    f_keys = feature_keys or FEATURE_KEYS
    s_keys = scalar_keys or SCALAR_KEYS
    idx_map: List[Tuple[str, int]] = []

    for k in f_keys:
        arr = entry[k]
        n_layers, dim = arr.shape
        for l in range(n_layers):
            idx_map.extend([(k, l)] * dim)

    for k in s_keys:
        idx_map.append((k, -1))
    return idx_map


def build_index_maps(
    entry: Dict[str, Any],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    f_keys = feature_keys or FEATURE_KEYS
    s_keys = scalar_keys or SCALAR_KEYS
    groups: List[str] = []
    layers: List[int] = []

    for k in f_keys:
        n_layers, dim = entry[k].shape
        for l in range(n_layers):
            groups.extend([k] * dim)
            layers.extend([l] * dim)

    for k in s_keys:
        groups.append(k)
        layers.append(-1)

    return np.array(groups), np.array(layers)


def extract_feature_matrix(
    dataset: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> np.ndarray:
    return np.stack([flatten_entry(e, feature_keys, scalar_keys) for e in dataset])


def detect_outliers_chisq(
    X_scaled: np.ndarray,
    confidence: float = OUTLIER_CONFIDENCE,
) -> Tuple[np.ndarray, float, np.ndarray]:
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    is_outlier, cutoff, sq_dist = detect_outliers_chisq(X_scaled, confidence=confidence)

    n_flagged = int(is_outlier.sum())
    print(f"[OutlierFilter] Confidence: {confidence*100:.1f}% | Cutoff distance²: {cutoff:.1f}")
    print(f"[OutlierFilter] Flagged {n_flagged} outlier(s) out of {len(dataset)} entries.")

    clean_mask = ~is_outlier
    X_clean = X[clean_mask]
    dataset_clean = [dataset[i] for i in range(len(dataset)) if clean_mask[i]]
    return X_clean, dataset_clean, is_outlier


def compute_classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_regression_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred)
    }


class ProbeTrainer:
    """
    Training, benchmarking, and export of linear probes and regression models
    on internal activation representations.
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
        X = extract_feature_matrix(dataset)

        if filter_extreme_outliers:
            X, dataset, _ = filter_outliers(X, dataset)

        relevance = np.array([
            e.get("response_political_relevance", e.get("meta", {}).get("severity", 0.0))
            for e in dataset
        ], dtype=np.float32)

        y_binary = (relevance > self.severity_threshold).astype(int)
        print(f"[ProbeTrainer] Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features.")
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
        pca_k = n_components or min(self.n_pca_components, X.shape[0] - 1)
        final_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_k, random_state=random_state)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])

        final_pipe.fit(X, y)
        joblib.dump(final_pipe, output_path)
        return final_pipe

    @staticmethod
    def load_classifier(filepath: str = DEFAULT_CLASSIFIER_PATH) -> Pipeline:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Classifier file not found at '{filepath}'.")
        return joblib.load(filepath)
