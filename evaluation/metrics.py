"""
Evaluation Metrics for Inpainting Quality.

Implements metrics computed on masked regions only:
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import torch
from PIL import Image

# Lazy imports for optional heavy dependencies
_lpips_model = None
_inception_model = None


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_val: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image (H, W, C) in [0, 1]
        target: Ground truth image (H, W, C) in [0, 1]
        mask: Optional mask (H, W) - compute only on masked region if provided
        max_val: Maximum possible value
        
    Returns:
        PSNR value in dB
    """
    if mask is not None:
        # Expand mask to 3D if needed
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        
        # Extract masked regions
        mask_bool = mask > 0.5
        pred_masked = pred[np.broadcast_to(mask_bool, pred.shape)]
        target_masked = target[np.broadcast_to(mask_bool, target.shape)]
        
        if len(pred_masked) == 0:
            return float('inf')
        
        mse = np.mean((pred_masked - target_masked) ** 2)
    else:
        mse = np.mean((pred - target) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    win_size: int = 7,
    data_range: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index.
    
    Args:
        pred: Predicted image (H, W, C) in [0, 1]
        target: Ground truth image (H, W, C) in [0, 1]
        mask: Optional mask (H, W) - weight SSIM by mask
        win_size: Window size for SSIM computation
        data_range: Data range of input images
        
    Returns:
        SSIM value [0, 1]
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")
    
    # Compute per-channel SSIM
    if pred.ndim == 3:
        ssim_vals = []
        for c in range(pred.shape[2]):
            s = ssim(pred[:, :, c], target[:, :, c],
                    win_size=win_size, data_range=data_range,
                    gaussian_weights=True)
            ssim_vals.append(s)
        ssim_value = np.mean(ssim_vals)
    else:
        ssim_value = ssim(pred, target, win_size=win_size, 
                         data_range=data_range, gaussian_weights=True)
    
    # If mask provided, we already computed on full image
    # Could weight by mask, but SSIM is computed in windows anyway
    # For masked-only, we'd need to crop to masked bounding box
    
    if mask is not None:
        # Compute on bounding box of mask
        mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
        rows = np.any(mask_2d > 0.5, axis=1)
        cols = np.any(mask_2d > 0.5, axis=0)
        
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            
            # Add padding for window
            pad = win_size
            r_min = max(0, r_min - pad)
            r_max = min(pred.shape[0], r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(pred.shape[1], c_max + pad)
            
            pred_crop = pred[r_min:r_max, c_min:c_max]
            target_crop = target[r_min:r_max, c_min:c_max]
            
            if pred_crop.shape[0] >= win_size and pred_crop.shape[1] >= win_size:
                if pred_crop.ndim == 3:
                    ssim_vals = []
                    for c in range(pred_crop.shape[2]):
                        s = ssim(pred_crop[:, :, c], target_crop[:, :, c],
                                win_size=win_size, data_range=data_range,
                                gaussian_weights=True)
                        ssim_vals.append(s)
                    ssim_value = np.mean(ssim_vals)
                else:
                    ssim_value = ssim(pred_crop, target_crop, 
                                     win_size=win_size, data_range=data_range,
                                     gaussian_weights=True)
    
    return float(ssim_value)


def compute_lpips(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    net: str = 'alex',
    device: Optional[str] = None
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity.
    
    Lower is better (measures perceptual difference).
    
    Args:
        pred: Predicted image (H, W, C) in [0, 1]
        target: Ground truth image (H, W, C) in [0, 1]
        mask: Optional mask - crops to masked bounding box
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Compute device
        
    Returns:
        LPIPS value (lower is better)
    """
    global _lpips_model
    
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips package required: pip install lpips")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model on first call
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net).to(device)
        _lpips_model.eval()
    
    # If mask provided, crop to bounding box
    if mask is not None:
        mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
        rows = np.any(mask_2d > 0.5, axis=1)
        cols = np.any(mask_2d > 0.5, axis=0)
        
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            
            # Add small padding
            pad = 16
            r_min = max(0, r_min - pad)
            r_max = min(pred.shape[0], r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(pred.shape[1], c_max + pad)
            
            pred = pred[r_min:r_max, c_min:c_max]
            target = target[r_min:r_max, c_min:c_max]
    
    # Convert to tensors: LPIPS expects (N, C, H, W) in [-1, 1]
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
    
    # Scale from [0, 1] to [-1, 1]
    pred_tensor = pred_tensor * 2 - 1
    target_tensor = target_tensor * 2 - 1
    
    pred_tensor = pred_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    with torch.no_grad():
        lpips_value = _lpips_model(pred_tensor, target_tensor)
    
    return float(lpips_value.item())


def compute_fid(
    pred_images: List[np.ndarray],
    target_images: List[np.ndarray],
    batch_size: int = 32,
    device: Optional[str] = None,
    use_masks: bool = False,
    masks: Optional[List[np.ndarray]] = None
) -> float:
    """
    Compute Fréchet Inception Distance between two sets of images.
    
    Lower is better.
    
    Args:
        pred_images: List of predicted images (H, W, C) in [0, 1]
        target_images: List of ground truth images
        batch_size: Batch size for Inception forward pass
        device: Compute device
        use_masks: If True and masks provided, crop images to mask regions
        masks: List of masks corresponding to images
        
    Returns:
        FID score
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        # Fallback to simple implementation
        return _compute_fid_simple(pred_images, target_images, device)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # If using masks, crop images to masked regions
    if use_masks and masks is not None:
        pred_images = [_crop_to_mask(img, mask) for img, mask in zip(pred_images, masks)]
        target_images = [_crop_to_mask(img, mask) for img, mask in zip(target_images, masks)]
    
    # Convert to tensors
    pred_tensors = []
    target_tensors = []
    
    for img in pred_images:
        # Resize to 299x299 for Inception
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((299, 299), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        pred_tensors.append(tensor)
    
    for img in target_images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((299, 299), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        target_tensors.append(tensor)
    
    pred_batch = torch.stack(pred_tensors)
    target_batch = torch.stack(target_tensors)
    
    # Get Inception features
    global _inception_model
    if _inception_model is None:
        _inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        _inception_model.eval()
    
    def get_features(batch):
        features = []
        with torch.no_grad():
            for i in range(0, len(batch), batch_size):
                batch_slice = batch[i:i+batch_size].to(device)
                feat = _inception_model(batch_slice)[0]
                feat = feat.squeeze(-1).squeeze(-1)
                features.append(feat.cpu())
        return torch.cat(features, dim=0).numpy()
    
    pred_features = get_features(pred_batch)
    target_features = get_features(target_batch)
    
    # Compute FID
    mu_pred = np.mean(pred_features, axis=0)
    mu_target = np.mean(target_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    sigma_target = np.cov(target_features, rowvar=False)
    
    from scipy import linalg
    
    diff = mu_pred - mu_target
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_pred @ sigma_target, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_pred + sigma_target - 2 * covmean)
    
    return float(fid)


def _compute_fid_simple(
    pred_images: List[np.ndarray],
    target_images: List[np.ndarray],
    device: Optional[str] = None
) -> float:
    """
    Simple FID approximation using features from a pretrained model.
    Falls back to this when pytorch-fid is not available.
    """
    try:
        import torchvision.models as models
        from torchvision import transforms
    except ImportError:
        # Return dummy value if no torch available
        return -1.0
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use pretrained VGG for features
    vgg = models.vgg16(pretrained=True).features[:16].to(device)
    vgg.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    def get_features(images):
        features = []
        with torch.no_grad():
            for img in images:
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                tensor = transform(tensor).to(device)
                feat = vgg(tensor)
                feat = feat.flatten(1).cpu().numpy()
                features.append(feat[0])
        return np.array(features)
    
    pred_features = get_features(pred_images)
    target_features = get_features(target_images)
    
    # Compute statistics
    mu_pred = np.mean(pred_features, axis=0)
    mu_target = np.mean(target_features, axis=0)
    
    if len(pred_features) < 2:
        return float(np.sum((mu_pred - mu_target) ** 2))
    
    sigma_pred = np.cov(pred_features, rowvar=False)
    sigma_target = np.cov(target_features, rowvar=False)
    
    from scipy import linalg
    
    diff = mu_pred - mu_target
    covmean, _ = linalg.sqrtm(sigma_pred @ sigma_target, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_pred + sigma_target - 2 * covmean)
    
    return float(fid)


def _crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    min_size: int = 64
) -> np.ndarray:
    """Crop image to bounding box of mask."""
    mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    rows = np.any(mask_2d > 0.5, axis=1)
    cols = np.any(mask_2d > 0.5, axis=0)
    
    if not rows.any() or not cols.any():
        return image
    
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    # Ensure minimum size
    height = r_max - r_min
    width = c_max - c_min
    
    if height < min_size:
        pad = (min_size - height) // 2
        r_min = max(0, r_min - pad)
        r_max = min(image.shape[0], r_max + pad)
    
    if width < min_size:
        pad = (min_size - width) // 2
        c_min = max(0, c_min - pad)
        c_max = min(image.shape[1], c_max + pad)
    
    return image[r_min:r_max, c_min:c_max]


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    compute_lpips: bool = True
) -> Dict[str, float]:
    """
    Compute all metrics for a single image pair.
    
    Args:
        pred: Predicted image (H, W, C) in [0, 1]
        target: Ground truth image (H, W, C) in [0, 1]
        mask: Optional mask (H, W) for masked-only metrics
        device: Compute device for LPIPS
        compute_lpips: Whether to compute LPIPS (slow)
        
    Returns:
        Dict with all metric values
    """
    metrics = {}
    
    # PSNR
    metrics['psnr'] = compute_psnr(pred, target, mask)
    
    # SSIM
    metrics['ssim'] = compute_ssim(pred, target, mask)
    
    # LPIPS
    if compute_lpips:
        try:
            metrics['lpips'] = globals()['compute_lpips'](pred, target, mask, device=device)
        except Exception as e:
            metrics['lpips'] = -1.0
            print(f"Warning: LPIPS computation failed: {e}")
    
    return metrics


class MetricsCalculator:
    """
    Class for computing metrics across multiple image pairs.
    Caches models and provides aggregation utilities.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        compute_fid: bool = True,
        compute_lpips: bool = True
    ):
        """
        Initialize metrics calculator.
        
        Args:
            device: Compute device
            compute_fid: Whether to compute FID
            compute_lpips: Whether to compute LPIPS
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.do_fid = compute_fid
        self.do_lpips = compute_lpips
        
        # Storage for FID computation
        self.pred_images: List[np.ndarray] = []
        self.target_images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        
        # Per-image metrics
        self.per_image_metrics: List[Dict[str, float]] = []
    
    def add_pair(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ):
        """
        Add an image pair for evaluation.
        
        Args:
            pred: Predicted image
            target: Ground truth image
            mask: Optional mask
        """
        # Compute per-image metrics
        metrics = compute_all_metrics(
            pred, target, mask,
            device=self.device,
            compute_lpips=self.do_lpips
        )
        self.per_image_metrics.append(metrics)
        
        # Store for FID
        if self.do_fid:
            self.pred_images.append(pred)
            self.target_images.append(target)
            if mask is not None:
                self.masks.append(mask)
    
    def compute_aggregate(self, use_masked_fid: bool = False) -> Dict[str, float]:
        """
        Compute aggregate metrics across all added pairs.
        
        Args:
            use_masked_fid: Compute FID on masked regions only
            
        Returns:
            Dict with mean metrics and FID
        """
        if len(self.per_image_metrics) == 0:
            return {}
        
        # Aggregate per-image metrics
        results = {}
        
        for key in self.per_image_metrics[0].keys():
            values = [m[key] for m in self.per_image_metrics if m[key] >= 0]
            if values:
                results[f'{key}_mean'] = float(np.mean(values))
                results[f'{key}_std'] = float(np.std(values))
        
        # Compute FID
        if self.do_fid and len(self.pred_images) >= 2:
            try:
                masks = self.masks if use_masked_fid and self.masks else None
                fid_value = compute_fid(
                    self.pred_images,
                    self.target_images,
                    device=self.device,
                    use_masks=use_masked_fid,
                    masks=masks
                )
                results['fid'] = fid_value
            except Exception as e:
                print(f"Warning: FID computation failed: {e}")
                results['fid'] = -1.0
        
        return results
    
    def reset(self):
        """Clear all stored data."""
        self.pred_images = []
        self.target_images = []
        self.masks = []
        self.per_image_metrics = []
    
    def get_per_image_results(self) -> List[Dict[str, float]]:
        """Get per-image metric values."""
        return self.per_image_metrics


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    methods: List[str] = None
) -> str:
    """
    Format metrics as a markdown/LaTeX-ready table.
    
    Args:
        results: Dict mapping method names to metric dicts
        methods: Order of methods (if None, uses results.keys())
        
    Returns:
        Formatted table string
    """
    if methods is None:
        methods = list(results.keys())
    
    # Get all metric names
    all_metrics = set()
    for method_results in results.values():
        all_metrics.update(method_results.keys())
    
    # Filter to main metrics
    main_metrics = ['fid', 'lpips_mean', 'psnr_mean', 'ssim_mean']
    metrics = [m for m in main_metrics if m in all_metrics]
    
    # Format header
    header = "| Method | " + " | ".join([m.replace('_mean', ' ↑' if 'psnr' in m or 'ssim' in m else ' ↓') for m in metrics]) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    
    # Format rows
    rows = []
    for method in methods:
        if method in results:
            row_vals = []
            for metric in metrics:
                val = results[method].get(metric, -1)
                if val >= 0:
                    row_vals.append(f"{val:.2f}")
                else:
                    row_vals.append("N/A")
            rows.append(f"| {method} | " + " | ".join(row_vals) + " |")
    
    table = "\n".join([header, separator] + rows)
    return table
