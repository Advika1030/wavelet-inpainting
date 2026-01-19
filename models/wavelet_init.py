"""
Core Wavelet-based Context Initialization for Diffusion Inpainting.

This module implements the adaptive frequency-aware blending approach:
- Low-frequency components (LL band) → trust context heavily (high alpha)
- Mid-frequency components (Level-3 details) → balanced trust
- High-frequency components (Level-1/2 details) → trust noise more (low alpha)

This preserves global coherence from context while allowing generative 
diversity in fine details.
"""

import numpy as np
import pywt
from typing import List, Tuple, Optional, Dict, Union
import torch
import torch.nn.functional as F
from scipy import ndimage


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample binary mask using max-pooling.
    
    CRITICAL: Uses max-pooling to ensure if ANY pixel in a window is masked,
    the entire block is considered masked. This is essential for correct
    wavelet coefficient masking.
    
    Args:
        mask: Binary mask (H, W) with values in {0, 1}
        factor: Downsampling factor (e.g., 2, 4, 8)
        
    Returns:
        Downsampled mask (H/factor, W/factor)
    """
    if factor == 1:
        return mask.copy()
    
    h, w = mask.shape
    new_h, new_w = h // factor, w // factor
    
    # Use max pooling - if any pixel in window is 1, output is 1
    # Reshape to blocks and take max
    mask_reshaped = mask[:new_h * factor, :new_w * factor]
    mask_reshaped = mask_reshaped.reshape(new_h, factor, new_w, factor)
    downsampled = mask_reshaped.max(axis=(1, 3))
    
    return downsampled.astype(np.float32)


def upsample_mask(mask: np.ndarray, factor: int, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample mask to target shape using nearest neighbor.
    
    Args:
        mask: Downsampled mask
        factor: Upsampling factor
        target_shape: (H, W) target dimensions
        
    Returns:
        Upsampled mask matching target_shape
    """
    # Simple repeat upsampling
    upsampled = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    
    # Crop or pad to exact target shape
    h, w = target_shape
    if upsampled.shape[0] > h:
        upsampled = upsampled[:h, :]
    if upsampled.shape[1] > w:
        upsampled = upsampled[:, :w]
    if upsampled.shape[0] < h or upsampled.shape[1] < w:
        padded = np.zeros(target_shape, dtype=upsampled.dtype)
        padded[:upsampled.shape[0], :upsampled.shape[1]] = upsampled
        upsampled = padded
        
    return upsampled


def blend_band(
    context_band: np.ndarray,
    noise_band: np.ndarray,
    mask: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Blend single wavelet sub-band (LL, LH, HL, or HH).
    
    Blending formula for masked regions:
        blended = alpha * context + (1 - alpha) * noise
        
    For unmasked regions, context is preserved.
    
    Args:
        context_band: Wavelet coefficients from context image
        noise_band: Wavelet coefficients from noise image
        mask: Binary mask at this decomposition level (1 = masked region)
        alpha: Blending weight (higher = more context, lower = more noise)
        
    Returns:
        Blended wavelet coefficients
    """
    # Ensure mask matches band shape
    if mask.shape != context_band.shape:
        # Resize mask to match band
        from scipy.ndimage import zoom
        zoom_factors = (context_band.shape[0] / mask.shape[0],
                       context_band.shape[1] / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)  # Nearest neighbor
        mask = mask[:context_band.shape[0], :context_band.shape[1]]
    
    # Blend: in masked region, mix context and noise; outside, keep context
    blended = (1 - mask) * context_band + mask * (alpha * context_band + (1 - alpha) * noise_band)
    
    return blended


def wavelet_initialize(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_schedule: List[float],
    wavelet: str = 'db4',
    level: int = 3,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None,
    boundary_smoothing: bool = True,
    smoothing_sigma: float = 2.0
) -> np.ndarray:
    """
    Perform wavelet-based adaptive context initialization.
    
    Core hypothesis: Different frequency bands should be blended with
    different trust levels:
    - Low frequencies (overall structure) → trust context (high alpha)
    - High frequencies (details) → allow noise for diversity (low alpha)
    
    Args:
        image: RGB image (H, W, 3) in [0, 1] range
        mask: Binary mask (H, W) where 1 = region to inpaint
        alpha_schedule: List of alpha values [alpha_LL, alpha_level3, alpha_level2/1]
                       Higher alpha = trust context more
        wavelet: Wavelet type (default 'db4' - Daubechies 4)
        level: Decomposition levels (default 3)
        noise_sigma: Standard deviation for Gaussian noise
        seed: Random seed for reproducibility
        boundary_smoothing: Apply Gaussian smoothing at mask boundaries
        smoothing_sigma: Sigma for boundary smoothing
        
    Returns:
        Initialized image (H, W, 3) with wavelet-blended masked region
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    assert image.ndim == 3 and image.shape[2] == 3, "Image must be (H, W, 3)"
    assert mask.ndim == 2, "Mask must be (H, W)"
    assert len(alpha_schedule) >= 3, "Alpha schedule needs at least 3 values"
    
    h, w, c = image.shape
    original_shape = (h, w)
    
    # Pad image to be divisible by 2^level for clean wavelet transform
    pad_h = (2**level - h % (2**level)) % (2**level)
    pad_w = (2**level - w % (2**level)) % (2**level)
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    padded_h, padded_w = image.shape[:2]
    
    # Generate noise image
    noise_image = np.random.randn(padded_h, padded_w, c) * noise_sigma
    
    # Process each channel independently
    initialized_channels = []
    
    for ch in range(c):
        channel = image[:, :, ch]
        noise_channel = noise_image[:, :, ch]
        
        # Perform wavelet decomposition on both context and noise
        context_coeffs = pywt.wavedec2(channel, wavelet, level=level)
        noise_coeffs = pywt.wavedec2(noise_channel, wavelet, level=level)
        
        # Blend coefficients at each level
        blended_coeffs = []
        
        # Process approximation (LL) band - use highest alpha (trust context)
        ll_context = context_coeffs[0]
        ll_noise = noise_coeffs[0]
        ll_mask = downsample_mask(mask, 2**level)
        
        # Ensure mask matches LL band size
        ll_mask_resized = _resize_mask_to_band(ll_mask, ll_context.shape)
        blended_ll = blend_band(ll_context, ll_noise, ll_mask_resized, alpha_schedule[0])
        blended_coeffs.append(blended_ll)
        
        # Process detail bands at each level
        for lv in range(1, level + 1):
            detail_tuple = context_coeffs[lv]  # (LH, HL, HH)
            noise_detail_tuple = noise_coeffs[lv]
            
            # Determine alpha for this level
            # Level index: 1 = highest level (coarsest), level = lowest level (finest)
            if lv == 1:
                # Highest level details (coarsest) - use second alpha
                alpha = alpha_schedule[1]
            else:
                # Lower level details (finer) - use third alpha
                alpha = alpha_schedule[2]
            
            # Downsample mask for this level
            downsample_factor = 2**(level - lv + 1)
            level_mask = downsample_mask(mask, downsample_factor)
            
            # Blend each detail band (LH, HL, HH)
            blended_details = []
            for context_band, noise_band in zip(detail_tuple, noise_detail_tuple):
                band_mask = _resize_mask_to_band(level_mask, context_band.shape)
                blended = blend_band(context_band, noise_band, band_mask, alpha)
                blended_details.append(blended)
            
            blended_coeffs.append(tuple(blended_details))
        
        # Reconstruct channel using inverse wavelet transform
        reconstructed = pywt.waverec2(blended_coeffs, wavelet)
        
        # Handle size mismatch from reconstruction
        reconstructed = reconstructed[:padded_h, :padded_w]
        
        initialized_channels.append(reconstructed)
    
    # Stack channels
    initialized = np.stack(initialized_channels, axis=-1)
    
    # Optional boundary smoothing to reduce artifacts
    if boundary_smoothing:
        initialized = _apply_boundary_smoothing(
            initialized, image, mask, smoothing_sigma
        )
    
    # Crop to original size
    initialized = initialized[:original_shape[0], :original_shape[1], :]
    
    # Clip to valid range
    initialized = np.clip(initialized, 0, 1)
    
    # Ensure unmasked regions are exactly preserved
    mask_3d = mask[:original_shape[0], :original_shape[1], np.newaxis]
    original_cropped = image[:original_shape[0], :original_shape[1], :]
    initialized = (1 - mask_3d) * original_cropped + mask_3d * initialized
    
    return initialized.astype(np.float32)


def _resize_mask_to_band(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize mask to match wavelet band dimensions."""
    from scipy.ndimage import zoom
    
    if mask.shape == target_shape:
        return mask
    
    # Calculate zoom factors
    zoom_h = target_shape[0] / mask.shape[0]
    zoom_w = target_shape[1] / mask.shape[1]
    
    resized = zoom(mask, (zoom_h, zoom_w), order=0)  # Nearest neighbor
    
    # Ensure exact size match
    if resized.shape[0] > target_shape[0]:
        resized = resized[:target_shape[0], :]
    if resized.shape[1] > target_shape[1]:
        resized = resized[:, :target_shape[1]]
    
    return resized


def _apply_boundary_smoothing(
    initialized: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Apply Gaussian smoothing at mask boundaries to reduce artifacts.
    
    Args:
        initialized: Wavelet-initialized image
        original: Original context image
        mask: Binary mask
        sigma: Gaussian smoothing sigma
        
    Returns:
        Smoothed result
    """
    from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
    
    # Create boundary region
    kernel_size = int(sigma * 3)
    if kernel_size < 1:
        return initialized
    
    struct = np.ones((kernel_size, kernel_size))
    dilated = binary_dilation(mask, struct)
    eroded = binary_erosion(mask, struct)
    boundary = dilated.astype(float) - eroded.astype(float)
    boundary = np.clip(boundary, 0, 1)
    
    # Create smooth transition weight
    smooth_weight = gaussian_filter(mask.astype(float), sigma)
    
    # Blend at boundary
    result = initialized.copy()
    boundary_3d = boundary[:, :, np.newaxis]
    smooth_weight_3d = smooth_weight[:, :, np.newaxis]
    
    # In boundary region, blend based on smooth weight
    blended = smooth_weight_3d * initialized + (1 - smooth_weight_3d) * original
    result = (1 - boundary_3d) * initialized + boundary_3d * blended
    
    return result


def check_wavelet_coeffs(coeffs: List) -> Dict:
    """
    Debug helper: Print shapes and value ranges of all wavelet bands.
    
    Args:
        coeffs: Output from pywt.wavedec2
        
    Returns:
        Dict with coefficient statistics
    """
    stats = {}
    
    # Approximation (LL) band
    ll = coeffs[0]
    stats['LL'] = {
        'shape': ll.shape,
        'min': float(ll.min()),
        'max': float(ll.max()),
        'mean': float(ll.mean()),
        'std': float(ll.std())
    }
    print(f"LL band: shape={ll.shape}, range=[{ll.min():.4f}, {ll.max():.4f}], "
          f"mean={ll.mean():.4f}, std={ll.std():.4f}")
    
    # Detail bands at each level
    for level_idx, detail_tuple in enumerate(coeffs[1:], 1):
        level_stats = {}
        for band_idx, (band_name, band) in enumerate(zip(['LH', 'HL', 'HH'], detail_tuple)):
            key = f'Level{level_idx}_{band_name}'
            level_stats[band_name] = {
                'shape': band.shape,
                'min': float(band.min()),
                'max': float(band.max()),
                'mean': float(band.mean()),
                'std': float(band.std())
            }
            print(f"Level {level_idx} {band_name}: shape={band.shape}, "
                  f"range=[{band.min():.4f}, {band.max():.4f}], "
                  f"mean={band.mean():.4f}, std={band.std():.4f}")
        stats[f'Level{level_idx}'] = level_stats
    
    return stats


def visualize_blending_weights(
    mask: np.ndarray,
    alpha_schedule: List[float],
    level: int = 3
) -> Dict[str, np.ndarray]:
    """
    Debug helper: Visualize alpha weights spatially for each level.
    
    Args:
        mask: Binary mask (H, W)
        alpha_schedule: Alpha values [LL, Level3, Level2/1]
        level: Decomposition levels
        
    Returns:
        Dict mapping level names to weight visualization arrays
    """
    h, w = mask.shape
    visualizations = {}
    
    # LL band weights
    ll_mask = downsample_mask(mask, 2**level)
    ll_weights = (1 - ll_mask) * 1.0 + ll_mask * alpha_schedule[0]
    visualizations['LL'] = ll_weights
    
    # Detail level weights
    for lv in range(1, level + 1):
        downsample_factor = 2**(level - lv + 1)
        level_mask = downsample_mask(mask, downsample_factor)
        
        if lv == 1:
            alpha = alpha_schedule[1]
        else:
            alpha = alpha_schedule[2]
        
        level_weights = (1 - level_mask) * 1.0 + level_mask * alpha
        visualizations[f'Level{lv}_details'] = level_weights
    
    return visualizations


def compare_init_methods(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_schedule: List[float] = [0.9, 0.6, 0.3],
    wavelet: str = 'db4',
    level: int = 3,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Debug helper: Compare all initialization methods side-by-side.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        alpha_schedule: For wavelet method
        wavelet: Wavelet type
        level: Decomposition level
        seed: Random seed
        
    Returns:
        Dict with 'pure_noise', 'naive_blend', 'wavelet' initialized images
    """
    np.random.seed(seed)
    h, w, c = image.shape
    
    # Pure noise initialization
    noise = np.random.randn(h, w, c) * 0.5
    pure_noise = (1 - mask[:, :, np.newaxis]) * image + mask[:, :, np.newaxis] * noise
    pure_noise = np.clip(pure_noise, 0, 1)
    
    # Naive blend (uniform alpha = 0.5)
    naive_alpha = 0.5
    blended = naive_alpha * image + (1 - naive_alpha) * noise
    naive_blend = (1 - mask[:, :, np.newaxis]) * image + mask[:, :, np.newaxis] * blended
    naive_blend = np.clip(naive_blend, 0, 1)
    
    # Wavelet-based initialization
    wavelet_init = wavelet_initialize(
        image, mask, alpha_schedule, wavelet, level, 
        noise_sigma=0.5, seed=seed
    )
    
    return {
        'original': image,
        'mask': mask,
        'pure_noise': pure_noise.astype(np.float32),
        'naive_blend': naive_blend.astype(np.float32),
        'wavelet': wavelet_init
    }


# Torch-compatible versions for pipeline integration

def wavelet_initialize_torch(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha_schedule: List[float],
    wavelet: str = 'db4',
    level: int = 3,
    noise_sigma: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    PyTorch wrapper for wavelet initialization.
    
    Args:
        image: Image tensor (B, C, H, W) or (C, H, W) in [0, 1]
        mask: Mask tensor (B, 1, H, W) or (1, H, W) or (H, W)
        alpha_schedule: Alpha values for each level
        wavelet: Wavelet type
        level: Decomposition levels
        noise_sigma: Noise standard deviation
        seed: Random seed
        
    Returns:
        Initialized image tensor, same shape as input
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
    batch_size = image.shape[0]
    
    results = []
    for i in range(batch_size):
        # Convert to numpy
        img_np = image[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        mask_np = mask[i, 0].cpu().numpy()  # (H, W)
        
        # Apply wavelet initialization
        init_np = wavelet_initialize(
            img_np, mask_np, alpha_schedule, wavelet, level,
            noise_sigma, seed=seed if seed is None else seed + i
        )
        
        # Convert back to tensor
        init_tensor = torch.from_numpy(init_np.transpose(2, 0, 1)).float()
        results.append(init_tensor)
    
    result = torch.stack(results, dim=0).to(device)
    
    if single_image:
        result = result.squeeze(0)
    
    return result


def get_wavelet_decomposition(
    image: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> Tuple[List, Dict]:
    """
    Get wavelet decomposition of image for visualization.
    
    Args:
        image: Grayscale image (H, W) or RGB image (H, W, 3)
        wavelet: Wavelet type
        level: Decomposition levels
        
    Returns:
        Tuple of (coefficients, shapes_dict)
    """
    if image.ndim == 3:
        # Use luminance channel
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    shapes = {
        'LL': coeffs[0].shape,
    }
    for i, detail in enumerate(coeffs[1:], 1):
        shapes[f'Level{i}'] = {
            'LH': detail[0].shape,
            'HL': detail[1].shape,
            'HH': detail[2].shape
        }
    
    return coeffs, shapes
