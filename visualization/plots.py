"""
Visualization utilities for wavelet inpainting experiments.

Provides functions for:
- Visualizing wavelet decomposition
- Creating comparison grids
- Plotting metrics tables
- Saving result images
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pywt


def visualize_wavelet_decomposition(
    image: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = 'gray'
) -> Optional[np.ndarray]:
    """
    Visualize wavelet decomposition showing all sub-bands.
    
    Creates a grid showing:
    - LL (approximation) band
    - LH, HL, HH (detail) bands at each level
    
    Args:
        image: Input image (H, W, 3) or (H, W) in [0, 1]
        wavelet: Wavelet type
        level: Decomposition levels
        save_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap for visualization
        
    Returns:
        Figure as numpy array if save_path is None
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()
    
    # Perform decomposition
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    
    # Create figure
    fig, axes = plt.subplots(level + 1, 4, figsize=figsize)
    fig.suptitle(f'Wavelet Decomposition ({wavelet}, {level} levels)', fontsize=14)
    
    # Plot approximation (LL) band
    ll = coeffs[0]
    axes[0, 0].imshow(ll, cmap=cmap)
    axes[0, 0].set_title(f'LL (approx)\n{ll.shape}')
    axes[0, 0].axis('off')
    
    # Hide other cells in first row
    for j in range(1, 4):
        axes[0, j].axis('off')
        axes[0, j].set_visible(False)
    
    # Plot detail bands at each level
    band_names = ['LH (horizontal)', 'HL (vertical)', 'HH (diagonal)']
    for lv in range(1, level + 1):
        detail_tuple = coeffs[lv]
        for j, (band, name) in enumerate(zip(detail_tuple, band_names)):
            # Normalize for visualization
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
            axes[lv, j + 1].imshow(band_norm, cmap=cmap)
            axes[lv, j + 1].set_title(f'Level {lv} {name}\n{band.shape}')
            axes[lv, j + 1].axis('off')
        
        # Empty first column for levels > 0
        axes[lv, 0].axis('off')
        axes[lv, 0].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        # Convert figure to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def create_comparison_grid(
    original: np.ndarray,
    masked: np.ndarray,
    results: Dict[str, np.ndarray],
    ground_truth: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 4),
    show_mask_border: bool = True,
    border_color: str = 'red',
    border_width: int = 2,
    title: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Create side-by-side comparison grid of inpainting results.
    
    Args:
        original: Original image (H, W, 3)
        masked: Masked image (H, W, 3)
        results: Dict mapping method names to result images
        ground_truth: Optional ground truth (H, W, 3)
        mask: Binary mask (H, W) for highlighting
        save_path: Path to save figure
        figsize: Figure size
        show_mask_border: Draw border around masked region
        border_color: Color of mask border
        border_width: Width of border
        title: Optional overall title
        
    Returns:
        Figure as numpy array if save_path is None
    """
    # Determine number of columns
    n_methods = len(results)
    n_cols = 2 + n_methods + (1 if ground_truth is not None else 0)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    def add_mask_border(ax, mask, color, width):
        """Add border around masked region."""
        if mask is None:
            return
        
        # Find bounding box of mask
        mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
        rows = np.any(mask_2d > 0.5, axis=1)
        cols = np.any(mask_2d > 0.5, axis=0)
        
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            
            rect = Rectangle(
                (c_min, r_min), c_max - c_min, r_max - r_min,
                linewidth=width, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
    
    col_idx = 0
    
    # Original image
    axes[col_idx].imshow(np.clip(original, 0, 1))
    axes[col_idx].set_title('Original')
    axes[col_idx].axis('off')
    if show_mask_border:
        add_mask_border(axes[col_idx], mask, border_color, border_width)
    col_idx += 1
    
    # Masked image
    axes[col_idx].imshow(np.clip(masked, 0, 1))
    axes[col_idx].set_title('Masked Input')
    axes[col_idx].axis('off')
    if show_mask_border:
        add_mask_border(axes[col_idx], mask, border_color, border_width)
    col_idx += 1
    
    # Method results
    for method_name, result in results.items():
        axes[col_idx].imshow(np.clip(result, 0, 1))
        axes[col_idx].set_title(method_name)
        axes[col_idx].axis('off')
        if show_mask_border:
            add_mask_border(axes[col_idx], mask, border_color, border_width)
        col_idx += 1
    
    # Ground truth
    if ground_truth is not None:
        axes[col_idx].imshow(np.clip(ground_truth, 0, 1))
        axes[col_idx].set_title('Ground Truth')
        axes[col_idx].axis('off')
        if show_mask_border:
            add_mask_border(axes[col_idx], mask, border_color, border_width)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def plot_metrics_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    highlight_best: bool = True
) -> Optional[np.ndarray]:
    """
    Generate visual table of metrics comparing methods.
    
    Args:
        results: Dict mapping method names to metric dicts
        save_path: Path to save figure
        figsize: Figure size
        highlight_best: Highlight best values
        
    Returns:
        Figure as numpy array if save_path is None
    """
    methods = list(results.keys())
    
    # Collect metrics
    metric_names = []
    for method_results in results.values():
        for key in method_results.keys():
            if key not in metric_names and not key.endswith('_std'):
                metric_names.append(key)
    
    # Determine which metrics should be higher (better) vs lower
    higher_better = {'psnr', 'ssim', 'psnr_mean', 'ssim_mean'}
    
    # Build table data
    cell_text = []
    for method in methods:
        row = []
        for metric in metric_names:
            val = results[method].get(metric, -1)
            if val >= 0:
                row.append(f"{val:.3f}")
            else:
                row.append("N/A")
        cell_text.append(row)
    
    # Find best values for highlighting
    best_indices = {}
    for j, metric in enumerate(metric_names):
        values = []
        for i, method in enumerate(methods):
            val = results[method].get(metric, -1)
            if val >= 0:
                values.append((val, i))
        
        if values:
            if any(m in metric.lower() for m in ['psnr', 'ssim']):
                # Higher is better
                best_idx = max(values, key=lambda x: x[0])[1]
            else:
                # Lower is better (FID, LPIPS)
                best_idx = min(values, key=lambda x: x[0])[1]
            best_indices[j] = best_idx
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Format column headers with arrows
    col_labels = []
    for metric in metric_names:
        if any(m in metric.lower() for m in ['psnr', 'ssim']):
            col_labels.append(f"{metric} ↑")
        elif any(m in metric.lower() for m in ['fid', 'lpips']):
            col_labels.append(f"{metric} ↓")
        else:
            col_labels.append(metric)
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight best values
    if highlight_best:
        for j, best_i in best_indices.items():
            cell = table[best_i + 1, j]  # +1 for header row
            cell.set_facecolor('#90EE90')  # Light green
    
    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', weight='bold')
    
    # Style row labels
    for i in range(len(methods)):
        cell = table[i + 1, -1]  # Row label column
        cell.set_facecolor('#D9E2F3')
    
    plt.title('Inpainting Quality Metrics Comparison', fontsize=12, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def save_result_image(
    image: np.ndarray,
    save_path: str,
    format: str = 'png'
):
    """
    Save image to file.
    
    Args:
        image: Image array (H, W, 3) in [0, 1] or (H, W)
        save_path: Output path
        format: Image format
    """
    from PIL import Image as PILImage
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Save
    if image.ndim == 2:
        pil_img = PILImage.fromarray(image, mode='L')
    else:
        pil_img = PILImage.fromarray(image, mode='RGB')
    
    pil_img.save(save_path, format=format.upper())


def create_initialization_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    init_results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> Optional[np.ndarray]:
    """
    Compare different initialization methods (before diffusion).
    
    Args:
        image: Original image
        mask: Binary mask
        init_results: Dict mapping method names to initialized images
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure as numpy array if save_path is None
    """
    n_methods = len(init_results)
    
    fig, axes = plt.subplots(1, n_methods + 2, figsize=figsize)
    
    # Original
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Masked
    mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    masked = image * (1 - mask_3d)
    axes[1].imshow(np.clip(masked, 0, 1))
    axes[1].set_title('Masked')
    axes[1].axis('off')
    
    # Initializations
    for i, (name, init_img) in enumerate(init_results.items()):
        axes[i + 2].imshow(np.clip(init_img, 0, 1))
        axes[i + 2].set_title(f'{name} Init')
        axes[i + 2].axis('off')
    
    plt.suptitle('Initialization Methods Comparison (Before Diffusion)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def plot_alpha_schedule_visualization(
    alpha_schedule: List[float],
    level: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[np.ndarray]:
    """
    Visualize alpha schedule across frequency levels.
    
    Args:
        alpha_schedule: [alpha_LL, alpha_level3, alpha_level2/1]
        level: Number of decomposition levels
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure as numpy array if save_path is None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart of alphas
    labels = ['LL (Low Freq)', f'Level {level} (Mid)', f'Levels 1-{level-1} (High)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    ax1.bar(labels, alpha_schedule, color=colors, edgecolor='black')
    ax1.set_ylabel('Alpha (Context Weight)')
    ax1.set_title('Alpha Schedule by Frequency Band')
    ax1.set_ylim(0, 1)
    
    for i, v in enumerate(alpha_schedule):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Explanation diagram
    ax2.axis('off')
    
    explanation = """
    Alpha Schedule Interpretation:
    
    α = 1.0  →  Pure context (no noise)
    α = 0.5  →  Equal mix
    α = 0.0  →  Pure noise (no context)
    
    Low frequencies (LL):
      Structure, lighting, color
      → High α (trust context)
    
    Mid frequencies (Level 3):
      Edges, major features
      → Balanced α
    
    High frequencies (Levels 1-2):
      Fine details, texture
      → Low α (generative diversity)
    """
    
    ax2.text(0.1, 0.5, explanation, transform=ax2.transAxes,
             fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Wavelet-Based Adaptive Blending', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array
