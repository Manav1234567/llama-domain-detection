"""
Utilities for feature representation transformation, dimensionality flattening,
and index mapping for mechanistic activations.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rep_guardrails.config import FEATURE_KEYS, SCALAR_KEYS


def flatten_entry(
    entry: Dict[str, Any],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Flattens multi-layer activation tensors and output scalar statistics into a unified 1D vector.
    
    Order of concatenation:
      1. Layer-wise multidimensional features (flattened) in order of FEATURE_KEYS.
      2. Sequence-level scalar statistics in order of SCALAR_KEYS.
      
    Args:
        entry: Dictionary containing extracted layer metrics and scalars.
        feature_keys: List of layer tensor keys (defaults to config.FEATURE_KEYS).
        scalar_keys: List of scalar statistic keys (defaults to config.SCALAR_KEYS).
        
    Returns:
        1D numpy array of type float32.
    """
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
    """
    Builds an index map mapping each column index in the flattened feature vector
    back to its origin feature group and layer index.
    
    Args:
        entry: Sample dataset entry containing representative shapes.
        feature_keys: List of layer tensor keys.
        scalar_keys: List of scalar statistic keys.
        
    Returns:
        List of (feature_group, layer_index) tuples. Non-layer scalars are assigned layer -1.
    """
    f_keys = feature_keys or FEATURE_KEYS
    s_keys = scalar_keys or SCALAR_KEYS
    idx_map: List[Tuple[str, int]] = []

    for k in f_keys:
        arr = entry[k]  # Shape [n_layers, dim]
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
    """
    Generates parallel numpy arrays identifying feature group name and layer index
    for every column of the feature matrix X.
    
    This is utilized for grouped permutation importance calculations to assess
    which conceptual groups (e.g. attention entropy vs residual stream) or
    transformer layers contribute most strongly to detection.
    
    Args:
        entry: Sample dataset entry containing representative feature shapes.
        feature_keys: List of layer tensor keys.
        scalar_keys: List of scalar keys.
        
    Returns:
        Tuple of (group_labels, layer_labels) as numpy arrays of length X.shape[1].
    """
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
        layers.append(-1)  # -1 represents token-level output scalar metrics

    return np.array(groups), np.array(layers)


def extract_feature_matrix(
    dataset: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None,
    scalar_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extracts a 2D numpy feature matrix X of shape (num_samples, feature_dim)
    from a list of processed dataset entries.
    
    Args:
        dataset: List of dataset entries.
        feature_keys: List of layer tensor keys.
        scalar_keys: List of scalar keys.
        
    Returns:
        2D numpy array of shape (N, D).
    """
    return np.stack([flatten_entry(e, feature_keys, scalar_keys) for e in dataset])
