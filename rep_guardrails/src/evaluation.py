import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_layer_separability(layer_idx, pos_activations, neg_activations, metric_name="PCA Component 1"):
    """
    Plots the separability of positive vs negative activations at a given layer.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(pos_activations, label="Positive (Political)", fill=True, color="firebrick")
    sns.kdeplot(neg_activations, label="Negative (Unrelated)", fill=True, color="seagreen")
    plt.title(f"Layer {layer_idx} Separability - {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_perplexity(model, text):
    """
    Dummy perplexity metric.
    """
    pass

def plot_feature_importance(importance_dict, title="Feature Importance"):
    """
    Plots a bar chart of feature importances.
    """
    keys = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values, color="steelblue")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
