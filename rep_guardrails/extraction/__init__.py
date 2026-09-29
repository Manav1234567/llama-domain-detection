"""
Feature extraction and manipulation subpackage.
"""

from rep_guardrails.extraction.feature_extractor import (
    PolygraphFeatureExtractor,
    UnifiedFeatureExtractor,
    LocalFeatureExtractor,
)
from rep_guardrails.extraction.feature_utils import (
    flatten_entry,
    get_feature_index_map,
    build_index_maps,
    extract_feature_matrix,
)

__all__ = [
    "PolygraphFeatureExtractor",
    "UnifiedFeatureExtractor",
    "LocalFeatureExtractor",
    "flatten_entry",
    "get_feature_index_map",
    "build_index_maps",
    "extract_feature_matrix",
]
