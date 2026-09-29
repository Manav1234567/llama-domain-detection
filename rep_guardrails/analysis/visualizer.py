"""
Visualization and interpretability diagnostics for internal representations,
PCA latent spaces, and layer-wise feature importance.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class LatentVisualizer:
    """
    Diagnostic visualizer for representation engineering, PCA manifolds,
    and feature/layer importance distributions.
    """

    @staticmethod
    def plot_pca_projection(
        X_scaled: np.ndarray,
        relevance: np.ndarray,
        domains: Optional[np.ndarray] = None,
        title: str = "PCA of Mechanistic Features",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> PCA:
        """
        Projects high-dimensional feature activations to 2D PCA space and creates
        a scatter plot colored by continuous domain relevance.
        
        Args:
            X_scaled: Standardized feature matrix.
            relevance: Continuous relevance scores (0.0 to 1.0).
            domains: Optional array of domain labels ('political', 'unrelated').
            title: Plot title.
            save_path: Optional file destination to save figure.
            show: If True, calls plt.show().
            
        Returns:
            Fitted PCA instance.
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        var_ratios = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 6))

        if domains is not None:
            is_unrelated = (domains == "unrelated")
            sc1 = ax.scatter(
                X_pca[~is_unrelated, 0],
                X_pca[~is_unrelated, 1],
                c=relevance[~is_unrelated],
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1,
                s=120,
                edgecolor="black",
                marker="o",
                label="Political",
            )
            sc2 = ax.scatter(
                X_pca[is_unrelated, 0],
                X_pca[is_unrelated, 1],
                c=relevance[is_unrelated],
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1,
                s=120,
                edgecolor="black",
                marker="X",
                label="Unrelated (Control)",
            )
            ax.legend()
        else:
            sc1 = ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=relevance,
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1,
                s=120,
                edgecolor="black",
            )

        cbar = plt.colorbar(sc1, ax=ax)
        cbar.set_label("Political Relevance (Score)", fontsize=11)

        ax.set_xlabel(f"PC1 ({var_ratios[0]*100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_ratios[1]*100:.1f}% variance)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Visualizer] Saved PCA projection plot to '{save_path}'.")

        if show:
            plt.show()
        else:
            plt.close()

        return pca

    @staticmethod
    def plot_loadings_importance(
        pca: PCA,
        idx_map: List[Tuple[str, int]],
        title_prefix: str = "Variance-Weighted Loadings",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes and plots variance-weighted PC1 and PC2 feature loadings grouped
        by conceptual feature type and by transformer layer.
        
        Args:
            pca: Fitted PCA instance (with at least 2 components).
            idx_map: Mapping of feature vector index to (group, layer).
            title_prefix: Prefix for chart headers.
            save_path: Optional image save path.
            show: If True, calls plt.show().
            
        Returns:
            Tuple of (group_importance_series, layer_importance_series).
        """
        df_imp = pd.DataFrame(idx_map, columns=["feature_group", "layer"])
        df_imp["pc1_loading"] = np.abs(pca.components_[0])
        df_imp["pc2_loading"] = np.abs(pca.components_[1])
        df_imp["combined"] = (
            df_imp["pc1_loading"] * pca.explained_variance_ratio_[0]
            + df_imp["pc2_loading"] * pca.explained_variance_ratio_[1]
        )

        group_importance = df_imp.groupby("feature_group")["combined"].sum().sort_values(ascending=False)
        layer_importance = df_imp[df_imp["layer"] >= 0].groupby("layer")["combined"].sum().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        group_importance.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
        axes[0].set_title(f"{title_prefix}: By Feature Group", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Summed |Loading| (Variance-Weighted)")
        axes[0].tick_params(axis="x", rotation=45)

        layer_importance.sort_index().plot(kind="bar", ax=axes[1], color="darkorange", edgecolor="black")
        axes[1].set_title(f"{title_prefix}: By Transformer Layer", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Layer Index (0 to N-1)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Visualizer] Saved loadings plot to '{save_path}'.")

        if show:
            plt.show()
        else:
            plt.close()

        return group_importance, layer_importance

    @staticmethod
    def plot_permutation_importances(
        group_imp: pd.Series,
        layer_imp: pd.Series,
        metric_name: str = "F1 Drop",
        title_prefix: str = "Permutation Importance",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plots permutation feature importance (metric drop when columns are shuffled)
        across feature groups and transformer layers.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        group_imp.plot(kind="bar", ax=axes[0], color="teal", edgecolor="black")
        axes[0].set_title(f"{title_prefix}: Feature Groups", fontsize=11, fontweight="bold")
        axes[0].set_ylabel(f"Metric Drop ({metric_name})")
        axes[0].tick_params(axis="x", rotation=45)

        layer_imp.sort_index().plot(kind="bar", ax=axes[1], color="firebrick", edgecolor="black")
        axes[1].set_title(f"{title_prefix}: Transformer Layers", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Layer Index")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Visualizer] Saved permutation importance plot to '{save_path}'.")

        if show:
            plt.show()
        else:
            plt.close()
