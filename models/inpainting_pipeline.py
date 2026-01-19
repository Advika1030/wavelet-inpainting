"""
Diffusion Inpainting Pipeline with Wavelet Initialization.

Integrates wavelet-based context initialization with Stable Diffusion inpainting.
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from PIL import Image
import torch
from tqdm import tqdm

# Lazy imports for optional dependencies
_diffusers_available = None
_pipeline = None


def _check_diffusers():
    """Check if diffusers is available and import pipeline."""
    global _diffusers_available, _pipeline
    if _diffusers_available is None:
        try:
            from diffusers import StableDiffusionInpaintPipeline
            _diffusers_available = True
            _pipeline = StableDiffusionInpaintPipeline
        except ImportError:
            _diffusers_available = False
            _pipeline = None
    return _diffusers_available


class WaveletInpaintingPipeline:
    """
    Complete inpainting pipeline combining wavelet initialization 
    with Stable Diffusion.
    
    Pipeline flow:
    1. Original Image + Mask
    2. → Wavelet-based initialization (frequency-aware blending)
    3. → Convert to PIL/Tensor format
    4. → Stable Diffusion Inpainting
    5. → Output Result
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_model: bool = True
    ):
        """
        Initialize the inpainting pipeline.
        
        Args:
            model_id: HuggingFace model ID for inpainting
            device: Device to run on ('cuda', 'cpu', or None for auto)
            torch_dtype: Data type (torch.float16 for speed, torch.float32 for quality)
            load_model: Whether to load the model immediately
        """
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == 'cuda' else torch.float32)
        
        self.sd_pipeline = None
        self._model_loaded = False
        
        if load_model:
            self.load_model()
    
    def load_model(self):
        """Load the Stable Diffusion inpainting model."""
        if self._model_loaded:
            return
        
        if not _check_diffusers():
            raise ImportError(
                "diffusers package is required. Install with: pip install diffusers"
            )
        
        print(f"Loading {self.model_id}...")
        
        try:
            self.sd_pipeline = _pipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,  # Disable for research
                requires_safety_checker=False
            )
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.sd_pipeline, 'enable_attention_slicing'):
                self.sd_pipeline.enable_attention_slicing()
            
            self._model_loaded = True
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.sd_pipeline is not None:
            del self.sd_pipeline
            self.sd_pipeline = None
            self._model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def initialize_with_wavelet(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        mask: Union[np.ndarray, torch.Tensor, Image.Image],
        alpha_schedule: List[float] = [0.9, 0.6, 0.3],
        wavelet: str = 'db4',
        level: int = 3,
        noise_sigma: float = 0.5,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply wavelet-based initialization to image.
        
        Args:
            image: Input image (RGB)
            mask: Binary mask (1 = region to inpaint)
            alpha_schedule: Blending weights per frequency level
            wavelet: Wavelet type
            level: Decomposition levels
            noise_sigma: Noise standard deviation
            seed: Random seed
            
        Returns:
            Initialized image as numpy array (H, W, 3)
        """
        from .wavelet_init import wavelet_initialize
        
        # Convert inputs to numpy
        image_np = self._to_numpy(image)
        mask_np = self._to_numpy_mask(mask)
        
        # Apply wavelet initialization
        initialized = wavelet_initialize(
            image_np, mask_np, alpha_schedule, wavelet, level,
            noise_sigma, seed
        )
        
        return initialized
    
    def inpaint(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        mask: Union[np.ndarray, torch.Tensor, Image.Image],
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        use_wavelet_init: bool = True,
        alpha_schedule: List[float] = [0.9, 0.6, 0.3],
        wavelet: str = 'db4',
        level: int = 3,
        noise_sigma: float = 0.5,
        seed: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform full inpainting with optional wavelet initialization.
        
        Args:
            image: Input image (RGB)
            mask: Binary mask (1 = region to inpaint)
            prompt: Text prompt for generation (optional for faces)
            negative_prompt: Negative prompt
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            use_wavelet_init: Whether to use wavelet initialization
            alpha_schedule: Blending weights for wavelet init
            wavelet: Wavelet type
            level: Decomposition levels
            noise_sigma: Noise standard deviation
            seed: Random seed
            return_intermediate: Return dict with all intermediate results
            
        Returns:
            Inpainted image or dict with intermediate results
        """
        if not self._model_loaded:
            self.load_model()
        
        # Set random seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            np.random.seed(seed)
        else:
            generator = None
        
        # Convert inputs
        image_np = self._to_numpy(image)
        mask_np = self._to_numpy_mask(mask)
        
        # Store original for metrics
        original_image = image_np.copy()
        
        # Apply wavelet initialization if enabled
        if use_wavelet_init:
            initialized = self.initialize_with_wavelet(
                image_np, mask_np, alpha_schedule, wavelet, level,
                noise_sigma, seed
            )
        else:
            initialized = image_np.copy()
        
        # Convert to PIL for Stable Diffusion pipeline
        # SD inpainting expects 512x512, so we need to resize
        original_size = image_np.shape[:2]
        target_size = (512, 512)  # SD inpainting standard size
        
        image_pil = self._numpy_to_pil(initialized, target_size)
        mask_pil = self._mask_to_pil(mask_np, target_size)
        
        # Run Stable Diffusion inpainting
        output = self.sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Get result and resize back to original
        result_pil = output.images[0]
        result_np = self._pil_to_numpy(result_pil, original_size)
        
        # Ensure unmasked regions match original
        result_np = (1 - mask_np[:, :, np.newaxis]) * original_image + \
                    mask_np[:, :, np.newaxis] * result_np
        result_np = np.clip(result_np, 0, 1).astype(np.float32)
        
        if return_intermediate:
            return {
                'original': original_image,
                'mask': mask_np,
                'initialized': initialized,
                'result': result_np
            }
        
        return result_np
    
    def inpaint_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        masks: List[Union[np.ndarray, Image.Image]],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Process batch of images.
        
        Args:
            images: List of input images
            masks: List of masks
            **kwargs: Arguments passed to inpaint()
            
        Returns:
            List of inpainted images
        """
        results = []
        seed = kwargs.get('seed')
        
        for i, (img, mask) in enumerate(tqdm(zip(images, masks), 
                                             total=len(images),
                                             desc="Inpainting")):
            # Increment seed for each image if provided
            if seed is not None:
                kwargs['seed'] = seed + i
            
            result = self.inpaint(img, mask, **kwargs)
            results.append(result)
        
        return results
    
    def _to_numpy(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """Convert image to numpy array (H, W, 3) in [0, 1]."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB')) / 255.0
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            if image.shape[0] == 3:  # (C, H, W)
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Ensure [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        return image.astype(np.float32)
    
    def _to_numpy_mask(self, mask: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """Convert mask to numpy array (H, W) with values in {0, 1}."""
        if isinstance(mask, Image.Image):
            mask = np.array(mask.convert('L')) / 255.0
        elif isinstance(mask, torch.Tensor):
            if mask.dim() == 4:
                mask = mask[0]
            if mask.dim() == 3:
                mask = mask[0]  # (H, W)
            mask = mask.cpu().numpy()
        
        # Ensure binary
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def _numpy_to_pil(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Convert numpy image to PIL, optionally resizing."""
        # Ensure [0, 255] uint8
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8, mode='RGB')
        
        if target_size is not None:
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
        
        return pil_image
    
    def _mask_to_pil(
        self,
        mask: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Convert mask to PIL, optionally resizing."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8, mode='L')
        
        if target_size is not None:
            pil_mask = pil_mask.resize(target_size, Image.NEAREST)
        
        return pil_mask
    
    def _pil_to_numpy(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Convert PIL to numpy, optionally resizing."""
        if target_size is not None:
            image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        
        return np.array(image).astype(np.float32) / 255.0


class MockInpaintingPipeline(WaveletInpaintingPipeline):
    """
    Mock pipeline for testing without loading large models.
    Replaces SD inpainting with simple noise-based filling.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['load_model'] = False
        super().__init__(*args, **kwargs)
        self._model_loaded = True  # Pretend model is loaded
    
    def load_model(self):
        """No-op for mock pipeline."""
        self._model_loaded = True
        print("Using mock pipeline (no SD model)")
    
    def inpaint(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        mask: Union[np.ndarray, torch.Tensor, Image.Image],
        use_wavelet_init: bool = True,
        seed: Optional[int] = None,
        return_intermediate: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Mock inpainting that just uses initialization result."""
        if seed is not None:
            np.random.seed(seed)
        
        image_np = self._to_numpy(image)
        mask_np = self._to_numpy_mask(mask)
        
        if use_wavelet_init:
            alpha_schedule = kwargs.get('alpha_schedule', [0.9, 0.6, 0.3])
            wavelet = kwargs.get('wavelet', 'db4')
            level = kwargs.get('level', 3)
            noise_sigma = kwargs.get('noise_sigma', 0.5)
            
            initialized = self.initialize_with_wavelet(
                image_np, mask_np, alpha_schedule, wavelet, level,
                noise_sigma, seed
            )
        else:
            initialized = image_np.copy()
        
        # Mock "diffusion" - just add slight blur and noise to initialized region
        from scipy.ndimage import gaussian_filter
        
        result = initialized.copy()
        for c in range(3):
            # Apply slight Gaussian blur to masked region
            blurred = gaussian_filter(result[:, :, c], sigma=1.0)
            result[:, :, c] = (1 - mask_np) * result[:, :, c] + mask_np * blurred
        
        # Add small amount of noise
        noise = np.random.randn(*result.shape) * 0.02
        result = result + mask_np[:, :, np.newaxis] * noise
        
        # Ensure unmasked region is preserved
        result = (1 - mask_np[:, :, np.newaxis]) * image_np + \
                 mask_np[:, :, np.newaxis] * result
        result = np.clip(result, 0, 1).astype(np.float32)
        
        if return_intermediate:
            return {
                'original': image_np,
                'mask': mask_np,
                'initialized': initialized,
                'result': result
            }
        
        return result


def create_pipeline(
    use_mock: bool = False,
    device: Optional[str] = None,
    **kwargs
) -> WaveletInpaintingPipeline:
    """
    Factory function to create appropriate pipeline.
    
    Args:
        use_mock: Use mock pipeline (for testing without GPU/models)
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Pipeline instance
    """
    if use_mock:
        return MockInpaintingPipeline(device=device, **kwargs)
    else:
        return WaveletInpaintingPipeline(device=device, **kwargs)
