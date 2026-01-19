"""
CelebA-HQ dataset loader for inpainting experiments.
Supports 1024×1024 resolution images with train/val/test splits.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .mask_generator import MaskGenerator, create_center_mask


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ Dataset for inpainting.
    
    Expected directory structure:
        data_root/
            celeba_hq/
                images/
                    00000.png
                    00001.png
                    ...
                    
    Or for HuggingFace format:
        data_root/
            celeba_hq/
                data/
                    train-00000-of-00001.parquet
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        resolution: int = 1024,
        mask_generator: Optional[MaskGenerator] = None,
        mask_config: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        return_dict: bool = True
    ):
        """
        Initialize CelebA-HQ dataset.
        
        Args:
            data_root: Root directory containing CelebA-HQ data
            split: One of 'train', 'val', 'test'
            resolution: Image resolution (typically 1024)
            mask_generator: MaskGenerator instance for creating masks
            mask_config: Configuration for mask generation
            transform: Optional additional transforms
            return_dict: If True, return dict; else return tuple
        """
        self.data_root = Path(data_root)
        self.split = split
        self.resolution = resolution
        self.return_dict = return_dict
        
        # Setup mask generator
        if mask_generator is None:
            mask_generator = MaskGenerator(resolution, resolution)
        self.mask_generator = mask_generator
        self.mask_config = mask_config or {'mask_type': 'center', 'mask_size': 256}
        
        # Setup transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), 
                            interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        self.additional_transform = transform
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        
        # Apply split
        self._apply_split()
        
        print(f"CelebA-HQ {split} split: {len(self.image_paths)} images")
        
    def _load_image_paths(self) -> List[Path]:
        """Load all image paths from data directory."""
        image_paths = []
        
        # Try different possible directory structures
        possible_dirs = [
            self.data_root / 'celeba_hq' / 'images',
            self.data_root / 'CelebA-HQ' / 'images',
            self.data_root / 'celeba-hq' / 'images',
            self.data_root / 'images',
            self.data_root,
        ]
        
        image_dir = None
        for d in possible_dirs:
            if d.exists():
                image_dir = d
                break
                
        if image_dir is None:
            # Create sample data directory for testing
            sample_dir = self.data_root / 'celeba_hq' / 'images'
            sample_dir.mkdir(parents=True, exist_ok=True)
            print(f"WARNING: No CelebA-HQ data found. Created directory: {sample_dir}")
            print("Please download CelebA-HQ dataset and place images in this directory.")
            print("Download instructions:")
            print("  1. From Kaggle: https://www.kaggle.com/datasets/lamsimon/celebahq")
            print("  2. From HuggingFace: https://huggingface.co/datasets/mattymchen/celeba-hq")
            print("  3. Original source: https://github.com/tkarras/progressive_growing_of_gans")
            return []
        
        # Collect image files
        extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
            
        # Sort for reproducibility
        image_paths = sorted(image_paths)
        
        return image_paths
    
    def _apply_split(self):
        """Apply train/val/test split (80/10/10)."""
        if len(self.image_paths) == 0:
            return
            
        n_total = len(self.image_paths)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        # Use fixed random state for reproducible splits
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        
        if self.split == 'train':
            split_indices = indices[:n_train]
        elif self.split == 'val':
            split_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            split_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        self.image_paths = [self.image_paths[i] for i in split_indices]
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[Dict, Tuple]:
        """
        Get item by index.
        
        Returns:
            Dict with keys: 'image', 'mask', 'masked_image', 'path'
            Or tuple: (image, mask, masked_image, path)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image_tensor = self.base_transform(image)
        if self.additional_transform is not None:
            image_tensor = self.additional_transform(image_tensor)
        
        # Generate mask
        mask = self.mask_generator.generate(**self.mask_config)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Create masked image (mask out region with zeros or noise)
        masked_image = image_tensor * (1 - mask_tensor)
        
        if self.return_dict:
            return {
                'image': image_tensor,          # (3, H, W) in [0, 1]
                'mask': mask_tensor,            # (1, H, W) binary
                'masked_image': masked_image,   # (3, H, W)
                'path': str(img_path)
            }
        else:
            return image_tensor, mask_tensor, masked_image, str(img_path)


class SyntheticCelebADataset(Dataset):
    """
    Synthetic dataset for testing when CelebA-HQ is not available.
    Generates random face-like images with gradients and patterns.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        resolution: int = 1024,
        mask_generator: Optional[MaskGenerator] = None,
        mask_config: Optional[Dict] = None,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.resolution = resolution
        self.mask_generator = mask_generator or MaskGenerator(resolution, resolution)
        self.mask_config = mask_config or {'mask_type': 'center', 'mask_size': 256}
        self.seed = seed
        
        print(f"Created synthetic dataset with {num_samples} samples")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        # Seeded random for reproducibility
        rng = np.random.RandomState(self.seed + idx)
        
        # Generate face-like image with smooth gradients
        h, w = self.resolution, self.resolution
        
        # Create smooth color gradients (face-like colors)
        base_color = rng.rand(3) * 0.3 + 0.4  # Skin-tone-ish
        
        # Create circular gradient (face shape)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        face_mask = np.clip(1 - distance / (min(h, w) * 0.45), 0, 1)
        
        # Create image
        image = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            # Add gradient
            gradient = np.linspace(0.3, 0.7, h).reshape(-1, 1)
            image[:, :, c] = base_color[c] * face_mask * gradient
            
            # Add some texture
            noise = rng.randn(h // 8, w // 8) * 0.02
            noise = np.repeat(np.repeat(noise, 8, axis=0), 8, axis=1)[:h, :w]
            image[:, :, c] += noise
            
        # Add "features" (simple shapes for eyes, nose, mouth)
        # Eyes
        cv2_avail = True
        try:
            import cv2
            cv2.circle(image, (center_x - 100, center_y - 80), 30, (0.2, 0.2, 0.3), -1)
            cv2.circle(image, (center_x + 100, center_y - 80), 30, (0.2, 0.2, 0.3), -1)
            # Nose
            cv2.ellipse(image, (center_x, center_y + 20), (20, 40), 0, 0, 360, 
                       (base_color[0] * 0.9, base_color[1] * 0.9, base_color[2] * 0.9), -1)
            # Mouth
            cv2.ellipse(image, (center_x, center_y + 120), (60, 20), 0, 0, 180, 
                       (0.6, 0.3, 0.3), -1)
        except:
            pass
            
        image = np.clip(image, 0, 1)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Generate mask
        mask = self.mask_generator.generate(**self.mask_config)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        masked_image = image_tensor * (1 - mask_tensor)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'masked_image': masked_image,
            'path': f'synthetic_{idx:05d}'
        }


def get_celeba_dataloaders(
    data_root: str,
    batch_size: int = 4,
    resolution: int = 1024,
    mask_config: Optional[Dict] = None,
    num_workers: int = 4,
    use_synthetic: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, test dataloaders for CelebA-HQ.
    
    Args:
        data_root: Root directory for data
        batch_size: Batch size
        resolution: Image resolution
        mask_config: Mask configuration
        num_workers: Number of data loading workers
        use_synthetic: Use synthetic data if real data not available
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    mask_gen = MaskGenerator(resolution, resolution, seed=42)
    mask_config = mask_config or {'mask_type': 'center', 'mask_size': 256}
    
    # Try to load real dataset
    train_dataset = CelebAHQDataset(
        data_root, 'train', resolution, mask_gen, mask_config
    )
    
    # If no real data found and synthetic fallback enabled
    if len(train_dataset) == 0 and use_synthetic:
        print("Using synthetic dataset for testing...")
        train_dataset = SyntheticCelebADataset(
            800, resolution, mask_gen, mask_config
        )
        val_dataset = SyntheticCelebADataset(
            100, resolution, mask_gen, mask_config
        )
        test_dataset = SyntheticCelebADataset(
            100, resolution, mask_gen, mask_config
        )
    else:
        val_dataset = CelebAHQDataset(
            data_root, 'val', resolution, mask_gen, mask_config
        )
        test_dataset = CelebAHQDataset(
            data_root, 'test', resolution, mask_gen, mask_config
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def download_celeba_hq_instructions() -> str:
    """Return instructions for downloading CelebA-HQ dataset."""
    instructions = """
    ================================================================
    CelebA-HQ Dataset Download Instructions
    ================================================================
    
    Option 1: HuggingFace (Recommended)
    ------------------------------------
    pip install datasets
    
    from datasets import load_dataset
    dataset = load_dataset("mattymchen/celeba-hq", split="train")
    
    # Save images to disk
    import os
    os.makedirs("data/celeba_hq/images", exist_ok=True)
    for i, item in enumerate(dataset):
        item['image'].save(f"data/celeba_hq/images/{i:05d}.png")
    
    Option 2: Kaggle
    -----------------
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials
    3. Download: kaggle datasets download -d lamsimon/celebahq
    4. Extract to data/celeba_hq/images/
    
    Option 3: Google Drive (Original)
    ----------------------------------
    1. Visit: https://github.com/tkarras/progressive_growing_of_gans
    2. Follow links to download CelebA-HQ
    3. Extract images to data/celeba_hq/images/
    
    Expected structure after download:
    data/
        celeba_hq/
            images/
                00000.png
                00001.png
                ...
                29999.png
    
    The dataset contains 30,000 high-quality celebrity face images at 1024x1024.
    ================================================================
    """
    return instructions
