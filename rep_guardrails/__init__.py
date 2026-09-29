"""
rep_guardrails: Representation Engineering Guardrails for Large Language Models.

A training-free mechanistic intervention framework that monitors internal LLM activations
(residual streams, attention entropy, and lookback ratios) to enforce deterministic,
inference-time guardrails and kill switches.
"""

__version__ = "0.1.0"

from rep_guardrails.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    FEATURE_KEYS,
    SCALAR_KEYS,
    SEVERITY_THRESHOLD,
    get_device,
    get_default_dtype,
)
from rep_guardrails.models.loader import (
    load_model_bundle,
    load_model_lite,
    show_architecture,
)
from rep_guardrails.extraction.feature_extractor import PolygraphFeatureExtractor
from rep_guardrails.extraction.feature_utils import (
    flatten_entry,
    get_feature_index_map,
    build_index_maps,
    extract_feature_matrix,
)
from rep_guardrails.data.corpus import (
    load_political_dataframe,
    load_existing_dataset,
    build_next_batch,
)
from rep_guardrails.data.synthetic import (
    build_domain_severity_corpus,
    build_mixed_corpus,
)
from rep_guardrails.data.judge import PoliticalJudge
from rep_guardrails.training.probe_trainer import ProbeTrainer
from rep_guardrails.training.outliers import detect_outliers_chisq, filter_outliers
from rep_guardrails.training.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    grouped_permutation_importance,
)
from rep_guardrails.analysis.visualizer import LatentVisualizer
from rep_guardrails.agents.guardrail_agent import GuardrailAgent, GuardrailResult
from rep_guardrails.agents.collection_agent import DataCollectionAgent
from rep_guardrails.agents.judge_agent import JudgeAgent
from rep_guardrails.agents.training_agent import ProbeTrainingAgent
from rep_guardrails.agents.analysis_agent import AnalysisAgent

__all__ = [
    "__version__",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_KEY",
    "FEATURE_KEYS",
    "SCALAR_KEYS",
    "SEVERITY_THRESHOLD",
    "get_device",
    "get_default_dtype",
    "load_model_bundle",
    "load_model_lite",
    "show_architecture",
    "PolygraphFeatureExtractor",
    "flatten_entry",
    "get_feature_index_map",
    "build_index_maps",
    "extract_feature_matrix",
    "load_political_dataframe",
    "load_existing_dataset",
    "build_next_batch",
    "build_domain_severity_corpus",
    "build_mixed_corpus",
    "PoliticalJudge",
    "ProbeTrainer",
    "detect_outliers_chisq",
    "filter_outliers",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "grouped_permutation_importance",
    "LatentVisualizer",
    "GuardrailAgent",
    "GuardrailResult",
    "DataCollectionAgent",
    "JudgeAgent",
    "ProbeTrainingAgent",
    "AnalysisAgent",
]
