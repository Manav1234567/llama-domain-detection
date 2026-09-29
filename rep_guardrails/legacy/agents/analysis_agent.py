"""
Analysis Agent for executing latent representation diagnostics,
PCA topological projection, and layer-importance visualization.
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler

from rep_guardrails.config import DEFAULT_DATASET_PATH
from rep_guardrails.data.corpus import load_existing_dataset
from rep_guardrails.extraction.feature_utils import extract_feature_matrix, get_feature_index_map
from rep_guardrails.training.outliers import filter_outliers
from rep_guardrails.analysis.visualizer import LatentVisualizer
from rep_guardrails.agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """
    Agent automating diagnostic analysis of latent geometries, PCA projections,
    and mechanistic feature distributions.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH) -> None:
        super().__init__(name="AnalysisAgent")
        self.dataset_path = dataset_path

    def analyze(
        self,
        save_pca_plot: Optional[str] = "latent_pca_projection.png",
        save_loadings_plot: Optional[str] = "feature_loadings_importance.png",
        filter_extreme_outliers: bool = True,
        show_plots: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete representation diagnostics and generates diagnostic figures.
        
        Args:
            save_pca_plot: File destination for PCA scatter plot.
            save_loadings_plot: File destination for loadings bar charts.
            filter_extreme_outliers: If True, filters Chi-Square outliers before PCA.
            show_plots: If True, invokes GUI display.
            
        Returns:
            Dictionary containing PCA explained variance and top feature/layer rankings.
        """
        self.log(f"Loading dataset from '{self.dataset_path}'...")
        dataset = load_existing_dataset(self.dataset_path)
        if not dataset:
            raise FileNotFoundError(f"No dataset found at '{self.dataset_path}'.")

        X = extract_feature_matrix(dataset)
        relevance = np.array([
            e.get("response_political_relevance", e.get("meta", {}).get("severity", 0.0))
            for e in dataset
        ], dtype=np.float32)
        domains = np.array([e.get("meta", {}).get("domain", "unspecified") for e in dataset])

        if filter_extreme_outliers:
            X, dataset, _ = filter_outliers(X, dataset)
            relevance = np.array([
                e.get("response_political_relevance", e.get("meta", {}).get("severity", 0.0))
                for e in dataset
            ], dtype=np.float32)
            domains = np.array([e.get("meta", {}).get("domain", "unspecified") for e in dataset])

        self.log(f"Standardizing {X.shape[0]} feature vectors of dimension {X.shape[1]}...")
        X_scaled = StandardScaler().fit_transform(X)

        self.log("Generating 2D PCA projection...")
        pca = LatentVisualizer.plot_pca_projection(
            X_scaled=X_scaled,
            relevance=relevance,
            domains=domains,
            title=f"PCA of Mechanistic Features (n={len(dataset)})",
            save_path=save_pca_plot,
            show=show_plots,
        )

        idx_map = get_feature_index_map(dataset[0])
        self.log("Computing variance-weighted loadings across groups and layers...")
        group_imp, layer_imp = LatentVisualizer.plot_loadings_importance(
            pca=pca,
            idx_map=idx_map,
            title_prefix="Variance-Weighted Loadings",
            save_path=save_loadings_plot,
            show=show_plots,
        )

        summary = {
            "num_samples": len(dataset),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "group_importance": group_imp.to_dict(),
            "top_layers": layer_imp.head(5).to_dict(),
        }

        self.log(f"PC1 Variance Explained: {pca.explained_variance_ratio_[0]*100:.2f}%")
        self.log(f"PC2 Variance Explained: {pca.explained_variance_ratio_[1]*100:.2f}%")
        self.log(f"Top 3 Feature Groups: {list(group_imp.head(3).index)}")
        self.log(f"Top 3 Transformer Layers: {list(layer_imp.head(3).index)}")

        return summary

    def run(self, save_plots: bool = True) -> Dict[str, Any]:
        """Agent entrypoint matching BaseAgent interface."""
        pca_path = "latent_pca_projection.png" if save_plots else None
        load_path = "feature_loadings_importance.png" if save_plots else None
        return self.analyze(save_pca_plot=pca_path, save_loadings_plot=load_path)
