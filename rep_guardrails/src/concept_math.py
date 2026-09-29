import torch
import numpy as np

def calculate_difference_in_means(activations_pos, activations_neg):
    """
    Calculates the difference-in-means vector (concept vector) from two sets of activations.
    
    Args:
        activations_pos (torch.Tensor or np.ndarray): Activations for positive/target concept. Shape: (N, D)
        activations_neg (torch.Tensor or np.ndarray): Activations for negative/baseline concept. Shape: (N, D)
        
    Returns:
        torch.Tensor or np.ndarray: Normalized difference-in-means concept vector.
    """
    if isinstance(activations_pos, np.ndarray):
        mean_pos = np.mean(activations_pos, axis=0)
        mean_neg = np.mean(activations_neg, axis=0)
        diff = mean_pos - mean_neg
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 0 else diff
    else:
        mean_pos = torch.mean(activations_pos, dim=0)
        mean_neg = torch.mean(activations_neg, dim=0)
        diff = mean_pos - mean_neg
        norm = torch.norm(diff)
        return diff / norm if norm > 0 else diff


def apply_orthogonal_projection(activation, concept_vector):
    """
    Projects the concept vector out of the activation (orthogonal projection).
    Used for the Erasure Kernel in Phase 2.
    
    Args:
        activation (torch.Tensor or np.ndarray): The original activation. Shape: (..., D)
        concept_vector (torch.Tensor or np.ndarray): The normalized concept vector. Shape: (D,)
        
    Returns:
        torch.Tensor or np.ndarray: The new activation with the concept erased.
    """
    if isinstance(activation, np.ndarray):
        # dot product along the last dimension
        proj_scale = np.dot(activation, concept_vector)
        # expand proj_scale to match activation shape if activation is 2D
        if activation.ndim == 2:
            proj_scale = proj_scale[:, np.newaxis]
        projection = proj_scale * concept_vector
        return activation - projection
    else:
        # dot product along the last dimension
        proj_scale = torch.matmul(activation, concept_vector)
        if activation.dim() == 2:
            proj_scale = proj_scale.unsqueeze(1)
        projection = proj_scale * concept_vector
        return activation - projection
