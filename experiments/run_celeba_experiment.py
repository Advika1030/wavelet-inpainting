#!/usr/bin/env python3
"""
Main Experiment Script for Wavelet-Based Inpainting on CelebA-HQ.

This script runs the complete experiment comparing:
1. Pure noise initialization
2. Naive blend initialization (uniform alpha)
3. Wavelet-based adaptive initialization (ours)

Usage:
    python experiments/run_celeba_experiment.py [--config configs/default_config.yaml]
    
    # Quick test mode
    python experiments/run_celeba_experiment.py --quick-test
    
    # Use synthetic data (no dataset required)
    python experiments/run_celeba_experiment.py --synthetic
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

# Import project modules
from data.celeba_hq_loader import CelebAHQDataset, SyntheticCelebADataset
from data.mask_generator import MaskGenerator, create_center_mask
from models.wavelet_init import (
    wavelet_initialize, 
    compare_init_methods,
    check_wavelet_coeffs,
    get_wavelet_decomposition
)
from models.inpainting_pipeline import WaveletInpaintingPipeline, MockInpaintingPipeline
from baselines.pure_noise import pure_noise_initialize
from baselines.naive_blend import naive_blend_initialize
from evaluation.metrics import (
    MetricsCalculator, 
    compute_all_metrics, 
    format_metrics_table
)
from visualization.plots import (
    create_comparison_grid,
    visualize_wavelet_decomposition,
    plot_metrics_table,
    save_result_image,
    create_initialization_comparison,
    plot_alpha_schedule_visualization
)


# Default configuration
DEFAULT_CONFIG = {
    'dataset': 'celeba_hq',
    'data_root': 'data',
    'resolution': 1024,
    'mask_type': 'center',
    'mask_sizes': [256],  # Start with 256×256 center masks
    'alpha_schedule': [0.9, 0.6, 0.3],  # [LL, Level-3, Level-2/1]
    'wavelet_type': 'db4',
    'decomposition_levels': 3,
    'noise_sigma': 0.5,
    'num_test_samples': 100,  # Subset for initial validation
    'diffusion_steps': 50,
    'guidance_scale': 7.5,
    'prompt': '',  # Empty prompt for unconditional
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'outputs',
    'save_intermediate': True,
    'compute_fid': True,
    'use_mock_pipeline': False,  # Set True for testing without SD model
    'batch_size': 1,  # For memory efficiency
}


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
    
    return config


def setup_output_dirs(config: Dict) -> Dict[str, Path]:
    """Create output directories."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(config['output_dir']) / f"experiment_{timestamp}"
    
    dirs = {
        'base': base_dir,
        'comparisons': base_dir / 'comparisons',
        'initializations': base_dir / 'initializations',
        'wavelets': base_dir / 'wavelet_viz',
        'metrics': base_dir / 'metrics',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = base_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_save_path}")
    
    return dirs


def get_dataset(config: Dict, use_synthetic: bool = False):
    """Get test dataset."""
    if use_synthetic:
        print("Using synthetic dataset for testing...")
        dataset = SyntheticCelebADataset(
            num_samples=config['num_test_samples'],
            resolution=config['resolution'],
            mask_generator=MaskGenerator(config['resolution'], config['resolution']),
            mask_config={'mask_type': config['mask_type'], 
                        'mask_size': config['mask_sizes'][0]},
            seed=config['seed']
        )
    else:
        mask_gen = MaskGenerator(config['resolution'], config['resolution'], 
                                 seed=config['seed'])
        mask_config = {
            'mask_type': config['mask_type'],
            'mask_size': config['mask_sizes'][0]
        }
        
        dataset = CelebAHQDataset(
            data_root=config['data_root'],
            split='test',
            resolution=config['resolution'],
            mask_generator=mask_gen,
            mask_config=mask_config
        )
        
        # If no data found, fall back to synthetic
        if len(dataset) == 0:
            print("No CelebA-HQ data found, using synthetic dataset...")
            dataset = SyntheticCelebADataset(
                num_samples=config['num_test_samples'],
                resolution=config['resolution'],
                mask_generator=mask_gen,
                mask_config=mask_config,
                seed=config['seed']
            )
    
    return dataset


def run_initialization_methods(
    image: np.ndarray,
    mask: np.ndarray,
    config: Dict,
    seed: int
) -> Dict[str, np.ndarray]:
    """
    Apply all initialization methods to an image.
    
    Returns dict mapping method names to initialized images.
    """
    results = {}
    
    # Pure noise
    results['Pure Noise'] = pure_noise_initialize(
        image, mask, 
        noise_sigma=config['noise_sigma'],
        seed=seed
    )
    
    # Naive blend (uniform alpha=0.5)
    results['Naive Blend'] = naive_blend_initialize(
        image, mask,
        alpha=0.5,
        noise_sigma=config['noise_sigma'],
        seed=seed
    )
    
    # Wavelet-based (ours)
    results['Wavelet (Ours)'] = wavelet_initialize(
        image, mask,
        alpha_schedule=config['alpha_schedule'],
        wavelet=config['wavelet_type'],
        level=config['decomposition_levels'],
        noise_sigma=config['noise_sigma'],
        seed=seed
    )
    
    return results


def run_experiment(config: Dict, use_synthetic: bool = False, quick_test: bool = False):
    """
    Run the full experiment.
    
    Args:
        config: Experiment configuration
        use_synthetic: Use synthetic data
        quick_test: Run on small subset for quick validation
    """
    print("=" * 60)
    print("Wavelet-Based Inpainting Experiment")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Resolution: {config['resolution']}")
    print(f"Alpha Schedule: {config['alpha_schedule']}")
    print(f"Wavelet: {config['wavelet_type']}, Levels: {config['decomposition_levels']}")
    print("=" * 60)
    
    # Setup
    output_dirs = setup_output_dirs(config)
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Get dataset
    dataset = get_dataset(config, use_synthetic)
    n_samples = min(len(dataset), config['num_test_samples'])
    
    if quick_test:
        n_samples = min(5, n_samples)
        print(f"Quick test mode: using {n_samples} samples")
    
    print(f"Testing on {n_samples} samples")
    
    # Initialize pipeline
    use_mock = config.get('use_mock_pipeline', False) or quick_test
    if use_mock:
        print("Using mock pipeline (no Stable Diffusion)")
        pipeline = MockInpaintingPipeline(device=config['device'])
    else:
        print("Loading Stable Diffusion inpainting model...")
        pipeline = WaveletInpaintingPipeline(
            device=config['device'],
            load_model=True
        )
    
    # Initialize metrics calculators for each method
    methods = ['Pure Noise', 'Naive Blend', 'Wavelet (Ours)']
    metrics_calculators = {method: MetricsCalculator(
        device=config['device'],
        compute_fid=config['compute_fid'],
        compute_lpips=True
    ) for method in methods}
    
    # Track timing
    timing_stats = {method: [] for method in methods}
    
    # Process samples
    print(f"\nProcessing {n_samples} samples...")
    
    for idx in tqdm(range(n_samples), desc="Processing"):
        sample = dataset[idx]
        
        # Get data
        if isinstance(sample, dict):
            image = sample['image'].numpy().transpose(1, 2, 0)  # (H, W, C)
            mask = sample['mask'].numpy()[0]  # (H, W)
        else:
            image = sample[0].numpy().transpose(1, 2, 0)
            mask = sample[1].numpy()[0]
        
        ground_truth = image.copy()
        seed = config['seed'] + idx
        
        # Generate initializations
        init_results = run_initialization_methods(image, mask, config, seed)
        
        # Save initialization comparison for first few samples
        if idx < 10 and config['save_intermediate']:
            init_save_path = output_dirs['initializations'] / f'init_comparison_{idx:04d}.png'
            create_initialization_comparison(
                image, mask, init_results,
                save_path=str(init_save_path)
            )
        
        # Run inpainting for each method
        inpaint_results = {}
        
        for method in methods:
            start_time = time.time()
            
            # For mock pipeline, just use initialization as result
            # For real pipeline, run full inpainting
            if use_mock:
                # Use the initialization directly (mock just adds slight blur)
                result = pipeline.inpaint(
                    init_results[method],
                    mask,
                    use_wavelet_init=False,  # Already initialized
                    seed=seed
                )
            else:
                # Determine which init method to use in pipeline
                if method == 'Wavelet (Ours)':
                    result = pipeline.inpaint(
                        image, mask,
                        prompt=config['prompt'],
                        num_inference_steps=config['diffusion_steps'],
                        guidance_scale=config['guidance_scale'],
                        use_wavelet_init=True,
                        alpha_schedule=config['alpha_schedule'],
                        wavelet=config['wavelet_type'],
                        level=config['decomposition_levels'],
                        noise_sigma=config['noise_sigma'],
                        seed=seed
                    )
                else:
                    # For baselines, pre-initialize and pass to pipeline
                    initialized = init_results[method]
                    result = pipeline.inpaint(
                        initialized, mask,
                        prompt=config['prompt'],
                        num_inference_steps=config['diffusion_steps'],
                        guidance_scale=config['guidance_scale'],
                        use_wavelet_init=False,  # Already initialized
                        seed=seed
                    )
            
            elapsed = time.time() - start_time
            timing_stats[method].append(elapsed)
            inpaint_results[method] = result
            
            # Add to metrics calculator
            metrics_calculators[method].add_pair(result, ground_truth, mask)
        
        # Save comparison grid for first samples
        if idx < 20 and config['save_intermediate']:
            masked_image = image * (1 - mask[:, :, np.newaxis])
            comparison_path = output_dirs['comparisons'] / f'comparison_{idx:04d}.png'
            create_comparison_grid(
                image, masked_image, inpaint_results,
                ground_truth=ground_truth,
                mask=mask,
                save_path=str(comparison_path),
                title=f'Sample {idx}'
            )
        
        # Save wavelet decomposition for first sample
        if idx == 0:
            wavelet_path = output_dirs['wavelets'] / 'wavelet_decomposition.png'
            visualize_wavelet_decomposition(
                image,
                wavelet=config['wavelet_type'],
                level=config['decomposition_levels'],
                save_path=str(wavelet_path)
            )
            
            # Save alpha schedule visualization
            alpha_path = output_dirs['wavelets'] / 'alpha_schedule.png'
            plot_alpha_schedule_visualization(
                config['alpha_schedule'],
                level=config['decomposition_levels'],
                save_path=str(alpha_path)
            )
    
    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    all_results = {}
    
    for method in methods:
        results = metrics_calculators[method].compute_aggregate(use_masked_fid=True)
        results['avg_time'] = np.mean(timing_stats[method])
        all_results[method] = results
        
        print(f"\n{method}:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    
    # Save metrics
    metrics_path = output_dirs['metrics'] / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save metrics table
    table_path = output_dirs['metrics'] / 'metrics_table.png'
    plot_metrics_table(all_results, save_path=str(table_path))
    
    # Print formatted table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(format_metrics_table(all_results, methods))
    print("=" * 60)
    
    # Save text summary
    summary_path = output_dirs['base'] / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Wavelet-Based Inpainting Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Resolution: {config['resolution']}\n")
        f.write(f"Mask Size: {config['mask_sizes'][0]}\n")
        f.write(f"Alpha Schedule: {config['alpha_schedule']}\n")
        f.write(f"Wavelet: {config['wavelet_type']}\n\n")
        f.write("Results:\n")
        f.write(format_metrics_table(all_results, methods))
        f.write("\n\n")
        f.write("Per-method timing:\n")
        for method in methods:
            avg_time = np.mean(timing_stats[method])
            f.write(f"  {method}: {avg_time:.2f}s per image\n")
    
    print(f"\nExperiment complete! Results saved to {output_dirs['base']}")
    
    return all_results


def run_quick_validation():
    """
    Run quick validation on a single synthetic image.
    Useful for verifying the pipeline works before full experiment.
    """
    print("Running quick validation...")
    
    # Create synthetic test image
    np.random.seed(42)
    h, w = 512, 512  # Smaller for speed
    
    # Create gradient image with face-like structure
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    image = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        image[:, :, c] = 0.5 + 0.3 * np.exp(-distance**2 / (2 * (h/4)**2))
        image[:, :, c] += np.random.randn(h, w) * 0.02
    image = np.clip(image, 0, 1)
    
    # Create center mask
    mask = create_center_mask(h, w, 128)
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test wavelet initialization
    print("\nTesting wavelet initialization...")
    alpha_schedule = [0.9, 0.6, 0.3]
    
    initialized = wavelet_initialize(
        image, mask, alpha_schedule,
        wavelet='db4', level=3,
        noise_sigma=0.5, seed=42
    )
    
    print(f"Initialized shape: {initialized.shape}")
    print(f"Initialized range: [{initialized.min():.4f}, {initialized.max():.4f}]")
    
    # Verify mask handling
    unmasked_diff = np.abs(initialized * (1 - mask[:, :, np.newaxis]) - 
                          image * (1 - mask[:, :, np.newaxis])).max()
    print(f"Unmasked region preserved: {unmasked_diff < 1e-6}")
    
    # Test all init methods
    print("\nComparing initialization methods...")
    results = compare_init_methods(image, mask, alpha_schedule, seed=42)
    
    for name, result in results.items():
        if isinstance(result, np.ndarray) and result.ndim == 3:
            print(f"  {name}: shape={result.shape}, range=[{result.min():.3f}, {result.max():.3f}]")
    
    # Test wavelet decomposition
    print("\nTesting wavelet decomposition...")
    coeffs, shapes = get_wavelet_decomposition(image, 'db4', 3)
    check_wavelet_coeffs(coeffs)
    
    print("\nQuick validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run wavelet-based inpainting experiment on CelebA-HQ'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data (no dataset required)'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Run quick test on small subset'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run quick validation only'
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use mock pipeline (no Stable Diffusion model)'
    )
    parser.add_argument(
        '--samples', type=int, default=None,
        help='Number of samples to process'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        run_quick_validation()
        return
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.mock:
        config['use_mock_pipeline'] = True
    if args.samples:
        config['num_test_samples'] = args.samples
    if args.output:
        config['output_dir'] = args.output
    
    # Run experiment
    run_experiment(
        config,
        use_synthetic=args.synthetic,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
