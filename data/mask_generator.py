"""
Mask generation utilities for inpainting experiments.
Supports center square masks, random rectangle masks, and irregular masks.
"""

import numpy as np
from typing import Tuple, Optional, List
import cv2


def create_center_mask(
    height: int, 
    width: int, 
    mask_size: int
) -> np.ndarray:
    """
    Create a center square mask.
    
    Args:
        height: Image height
        width: Image width
        mask_size: Size of the square mask (e.g., 128, 256, 512)
        
    Returns:
        Binary mask (H, W) where 1 indicates masked region
    """
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate center position
    center_y, center_x = height // 2, width // 2
    half_size = mask_size // 2
    
    # Set masked region to 1
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    
    mask[y_start:y_end, x_start:x_end] = 1.0
    
    return mask


def create_random_rectangle_mask(
    height: int, 
    width: int,
    min_size: int = 64,
    max_size: int = 256,
    num_rectangles: int = 1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create random rectangle masks.
    
    Args:
        height: Image height
        width: Image width
        min_size: Minimum rectangle dimension
        max_size: Maximum rectangle dimension
        num_rectangles: Number of rectangles to add
        seed: Random seed for reproducibility
        
    Returns:
        Binary mask (H, W) where 1 indicates masked region
    """
    if seed is not None:
        np.random.seed(seed)
        
    mask = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_rectangles):
        # Random size
        rect_h = np.random.randint(min_size, min(max_size, height))
        rect_w = np.random.randint(min_size, min(max_size, width))
        
        # Random position
        y = np.random.randint(0, height - rect_h)
        x = np.random.randint(0, width - rect_w)
        
        mask[y:y+rect_h, x:x+rect_w] = 1.0
    
    return mask


def create_irregular_mask(
    height: int,
    width: int,
    num_strokes: int = 5,
    stroke_width_range: Tuple[int, int] = (10, 40),
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create irregular brush-stroke mask.
    
    Args:
        height: Image height
        width: Image width
        num_strokes: Number of brush strokes
        stroke_width_range: (min, max) stroke width
        seed: Random seed for reproducibility
        
    Returns:
        Binary mask (H, W) where 1 indicates masked region
    """
    if seed is not None:
        np.random.seed(seed)
        
    mask = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_strokes):
        # Random starting point
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        
        # Random bezier-like stroke
        num_points = np.random.randint(3, 8)
        points = [(x1, y1)]
        
        for _ in range(num_points):
            # Move in random direction
            dx = np.random.randint(-100, 100)
            dy = np.random.randint(-100, 100)
            new_x = np.clip(points[-1][0] + dx, 0, width - 1)
            new_y = np.clip(points[-1][1] + dy, 0, height - 1)
            points.append((int(new_x), int(new_y)))
        
        # Draw polyline
        stroke_width = np.random.randint(*stroke_width_range)
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [points_array], False, 1.0, stroke_width)
        
        # Add circles at vertices for smoother strokes
        for point in points:
            cv2.circle(mask, point, stroke_width // 2, 1.0, -1)
    
    return mask


class MaskGenerator:
    """
    Unified mask generator supporting multiple mask types.
    """
    
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = 42
    ):
        """
        Initialize mask generator.
        
        Args:
            height: Default image height
            width: Default image width
            seed: Random seed for reproducibility
        """
        self.height = height
        self.width = width
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
    def reset_seed(self, seed: Optional[int] = None):
        """Reset random state."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.RandomState(self.seed)
        
    def generate(
        self,
        mask_type: str = 'center',
        **kwargs
    ) -> np.ndarray:
        """
        Generate mask of specified type.
        
        Args:
            mask_type: One of 'center', 'random_rect', 'irregular'
            **kwargs: Type-specific parameters
            
        Returns:
            Binary mask (H, W)
        """
        height = kwargs.pop('height', self.height)
        width = kwargs.pop('width', self.width)
        
        if mask_type == 'center':
            mask_size = kwargs.get('mask_size', 256)
            return create_center_mask(height, width, mask_size)
            
        elif mask_type == 'random_rect':
            return create_random_rectangle_mask(
                height, width,
                min_size=kwargs.get('min_size', 64),
                max_size=kwargs.get('max_size', 256),
                num_rectangles=kwargs.get('num_rectangles', 1),
                seed=self._rng.randint(0, 2**31)
            )
            
        elif mask_type == 'irregular':
            return create_irregular_mask(
                height, width,
                num_strokes=kwargs.get('num_strokes', 5),
                stroke_width_range=kwargs.get('stroke_width_range', (10, 40)),
                seed=self._rng.randint(0, 2**31)
            )
            
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
    
    def generate_batch(
        self,
        batch_size: int,
        mask_type: str = 'center',
        **kwargs
    ) -> np.ndarray:
        """
        Generate batch of masks.
        
        Args:
            batch_size: Number of masks to generate
            mask_type: Type of mask
            **kwargs: Type-specific parameters
            
        Returns:
            Batch of masks (B, H, W)
        """
        masks = []
        for _ in range(batch_size):
            mask = self.generate(mask_type, **kwargs)
            masks.append(mask)
        return np.stack(masks, axis=0)


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Dilate mask to create boundary padding.
    
    Args:
        mask: Binary mask (H, W)
        kernel_size: Dilation kernel size
        
    Returns:
        Dilated mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(np.float32)


def get_mask_boundary(mask: np.ndarray, boundary_width: int = 5) -> np.ndarray:
    """
    Extract boundary region of mask.
    
    Args:
        mask: Binary mask (H, W)
        boundary_width: Width of boundary region
        
    Returns:
        Boundary mask
    """
    dilated = dilate_mask(mask, boundary_width)
    eroded = cv2.erode(mask.astype(np.uint8), 
                       np.ones((boundary_width, boundary_width), np.uint8),
                       iterations=1)
    boundary = dilated - eroded.astype(np.float32)
    return np.clip(boundary, 0, 1)
