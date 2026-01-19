"""
Naive Blend Baseline for Inpainting Initialization.

This baseline uses uniform alpha blending between context and noise,
without any frequency-aware adaptation.
"""

import numpy as np
from typing import Optional, Union
import torch


def naive_blend_initialize(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize masked region with uniform alpha blending.
    
    This baseline blends context and noise uniformly across all pixels
    in the masked region, without considering frequency content.
    
    Blending formula for masked region:
        result = alpha * context + (1 - alpha) * noise
    
    Args:
        image: RGB image (H, W, 3) in [0, 1] range
        mask: Binary mask (H, W) where 1 = region to inpaint
        alpha: Blending weight (higher = more context, lower = more noise)
               Default 0.5 means equal mix
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
    assert 0 <= alpha <= 1, "Alpha must be in [0, 1]"
    
    h, w, c = image.shape
    
    # Generate Gaussian noise
    noise = np.random.randn(h, w, c) * noise_sigma
    
    # Blend context and noise with uniform alpha
    blended = alpha * image + (1 - alpha) * noise
    blended = np.clip(blended, 0, 1)
    
    # Apply blending to masked region only
    mask_3d = mask[:, :, np.newaxis]
    result = (1 - mask_3d) * image + mask_3d * blended
    
    return result.astype(np.float32)


def naive_blend_initialize_torch(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    PyTorch version of naive blend initialization.
    
    Args:
        image: Image tensor (B, C, H, W) or (C, H, W) in [0, 1]
        mask: Mask tensor (B, 1, H, W) or (1, H, W) or (H, W)
        alpha: Blending weight
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
    noise = torch.randn_like(image) * noise_sigma
    
    # Blend
    blended = alpha * image + (1 - alpha) * noise
    blended = torch.clamp(blended, 0, 1)
    
    # Apply to masked region
    result = (1 - mask) * image + mask * blended
    
    if single_image:
        result = result.squeeze(0)
    
    return result


def gradient_aware_blend_initialize(
    image: np.ndarray,
    mask: np.ndarray,
    base_alpha: float = 0.5,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Slight enhancement: vary alpha based on local gradient magnitude.
    
    Areas with higher gradients (edges) get less context weight,
    while flat areas get more. Still not as sophisticated as wavelet approach.
    
    Args:
        image: RGB image (H, W, 3) in [0, 1] range
        mask: Binary mask (H, W)
        base_alpha: Base blending weight
        noise_sigma: Noise standard deviation
        seed: Random seed
        
    Returns:
        Initialized image (H, W, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    from scipy.ndimage import sobel
    
    h, w, c = image.shape
    
    # Compute gradient magnitude (on grayscale)
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    grad_x = sobel(gray, axis=0)
    grad_y = sobel(gray, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    
    # Compute spatially varying alpha
    # High gradient = lower alpha (more noise), low gradient = higher alpha (more context)
    alpha_map = base_alpha + (0.3 - 0.6 * grad_mag)  # Range: [base-0.3, base+0.3]
    alpha_map = np.clip(alpha_map, 0.1, 0.9)
    
    # Generate noise
    noise = np.random.randn(h, w, c) * noise_sigma
    
    # Blend with spatially varying alpha
    alpha_3d = alpha_map[:, :, np.newaxis]
    blended = alpha_3d * image + (1 - alpha_3d) * noise
    blended = np.clip(blended, 0, 1)
    
    # Apply to masked region
    mask_3d = mask[:, :, np.newaxis]
    result = (1 - mask_3d) * image + mask_3d * blended
    
    return result.astype(np.float32)
