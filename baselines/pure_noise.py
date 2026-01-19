"""
Pure Noise Baseline for Inpainting Initialization.

This baseline initializes the masked region with pure Gaussian noise,
providing no context information from the surrounding image.
"""

import numpy as np
from typing import Optional, Union
import torch


def pure_noise_initialize(
    image: np.ndarray,
    mask: np.ndarray,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize masked region with pure Gaussian noise.
    
    This is the simplest baseline - no context information is used.
    The masked region is filled with random Gaussian noise.
    
    Args:
        image: RGB image (H, W, 3) in [0, 1] range
        mask: Binary mask (H, W) where 1 = region to inpaint
        noise_sigma: Standard deviation for Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        Initialized image (H, W, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    assert image.ndim == 3 and image.shape[2] == 3, "Image must be (H, W, 3)"
    assert mask.ndim == 2, "Mask must be (H, W)"
    
    h, w, c = image.shape
    
    # Generate pure Gaussian noise
    noise = np.random.randn(h, w, c) * noise_sigma
    
    # Clip to [0, 1] - noise can go negative so we shift it
    # Center noise around 0.5 for valid image values
    noise = noise + 0.5
    noise = np.clip(noise, 0, 1)
    
    # Apply noise to masked region only
    mask_3d = mask[:, :, np.newaxis]
    result = (1 - mask_3d) * image + mask_3d * noise
    
    return result.astype(np.float32)


def pure_noise_initialize_torch(
    image: torch.Tensor,
    mask: torch.Tensor,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    PyTorch version of pure noise initialization.
    
    Args:
        image: Image tensor (B, C, H, W) or (C, H, W) in [0, 1]
        mask: Mask tensor (B, 1, H, W) or (1, H, W) or (H, W)
        noise_sigma: Noise standard deviation
        seed: Random seed
        
    Returns:
        Initialized image tensor
    """
    # Handle batched input
    single_image = image.dim() == 3
    if single_image:
        image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
    
    device = image.device
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate noise
    noise = torch.randn_like(image) * noise_sigma + 0.5
    noise = torch.clamp(noise, 0, 1)
    
    # Blend
    result = (1 - mask) * image + mask * noise
    
    if single_image:
        result = result.squeeze(0)
    
    return result
