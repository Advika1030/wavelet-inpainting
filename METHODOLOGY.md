# Adaptive Wavelet-Based Context Initialization for Diffusion Inpainting

## Detailed Methodology Document

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Core Hypothesis](#3-core-hypothesis)
4. [Technical Approach](#4-technical-approach)
5. [Implementation Details](#5-implementation-details)
6. [Experimental Design](#6-experimental-design)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Project Architecture](#8-project-architecture)
9. [Reproducibility](#9-reproducibility)

---

## 1. Executive Summary

This research project proposes a novel **initialization strategy** for diffusion-based image inpainting. Instead of using uniform blending or pure noise to initialize masked regions before diffusion, we leverage **wavelet decomposition** to apply frequency-aware blending:

- **Low frequencies** (structure, color) → heavily trust context
- **High frequencies** (texture, details) → allow noise for generative diversity

This approach is:
- **Training-free**: Uses pre-trained Stable Diffusion without fine-tuning
- **Model-agnostic**: Can be applied to any diffusion inpainting model
- **Computationally efficient**: Adds minimal overhead (wavelet transform only)

---

## 2. Problem Statement

### 2.1 Background

Diffusion-based inpainting models (e.g., Stable Diffusion Inpainting) work by:
1. Taking an image with a masked region
2. Initializing the masked region with some content
3. Running the diffusion denoising process
4. Outputting the completed image

### 2.2 The Initialization Problem

The **initialization of the masked region** significantly affects results:

| Approach | Description | Problem |
|----------|-------------|---------|
| **Pure Noise** | Fill with random Gaussian noise | Loses all context; diffusion must "guess" everything |
| **Copy Context** | Extend surrounding pixels | Creates visible seams; lacks diversity |
| **Uniform Blend** | Mix context + noise uniformly | Treats all frequencies equally; suboptimal |

### 2.3 Our Insight

**Not all frequency components should be treated equally.** The surrounding context provides valuable information about:
- Overall lighting and color tone (low frequency) → should be trusted
- General structure and edges (mid frequency) → partially trusted
- Fine textures and details (high frequency) → let the model generate these

---

## 3. Core Hypothesis

> **Frequency-aware initialization using wavelet decomposition can improve diffusion inpainting quality by preserving global coherence from context while allowing generative diversity in fine details.**

### 3.1 Alpha Schedule

We define an **alpha schedule** that varies trust levels by frequency:

```
α_schedule = [α_LL, α_mid, α_high] = [0.9, 0.6, 0.3]
```

| Band | Alpha | Meaning |
|------|-------|---------|
| **LL (approximation)** | 0.9 | 90% context, 10% noise |
| **Level 3 details** | 0.6 | 60% context, 40% noise |
| **Level 1-2 details** | 0.3 | 30% context, 70% noise |

### 3.2 Blending Formula

For each wavelet band in the masked region:

```
blended = α × context_band + (1 - α) × noise_band
```

Outside the masked region, context is preserved exactly.

---

## 4. Technical Approach

### 4.1 Wavelet Decomposition

We use the **2D Discrete Wavelet Transform (DWT)** with the Daubechies-4 (db4) wavelet:

```
Image → DWT → [LL, (LH₁, HL₁, HH₁), (LH₂, HL₂, HH₂), (LH₃, HL₃, HH₃)]
              └─────────────────────────────────────────────────────────┘
                                    3-level decomposition
```

#### Sub-band Meanings:
- **LL**: Low-frequency approximation (1/8 resolution) - overall structure, average color
- **LH**: Horizontal details - horizontal edges
- **HL**: Vertical details - vertical edges  
- **HH**: Diagonal details - texture, noise, fine details

### 4.2 Algorithm

```python
def wavelet_initialize(image, mask, alpha_schedule, wavelet='db4', level=3):
    """
    1. Decompose context image into wavelet coefficients
    2. Generate noise image and decompose similarly
    3. For each band:
       - Downsample mask to match band resolution
       - Blend context and noise based on alpha for that frequency
    4. Reconstruct image using inverse wavelet transform
    5. Apply boundary smoothing to reduce artifacts
    """
```

### 4.3 Mask Handling

**Critical detail**: Masks must be downsampled correctly for each wavelet level.

We use **max-pooling** for mask downsampling:
- If ANY pixel in a block is masked → entire wavelet coefficient is treated as masked
- This prevents context leakage into masked regions

```python
def downsample_mask(mask, factor):
    # Reshape to blocks and take max
    mask_reshaped = mask.reshape(new_h, factor, new_w, factor)
    return mask_reshaped.max(axis=(1, 3))
```

### 4.4 Pipeline Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Image + Mask                                         │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────────────────────────────┐                   │
│   │   WAVELET INITIALIZATION (Ours)     │                   │
│   │   1. DWT decomposition              │                   │
│   │   2. Frequency-aware blending       │                   │
│   │   3. IDWT reconstruction            │                   │
│   └─────────────────────────────────────┘                   │
│         │                                                    │
│         ▼                                                    │
│   Initialized Image                                          │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────────────────────────────┐                   │
│   │   STABLE DIFFUSION INPAINTING       │                   │
│   │   (Pre-trained, frozen weights)     │                   │
│   │   - 50 diffusion steps              │                   │
│   │   - Guidance scale: 7.5             │                   │
│   └─────────────────────────────────────┘                   │
│         │                                                    │
│         ▼                                                    │
│   Final Inpainted Image                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Details

### 5.1 Wavelet Choice: Daubechies-4 (db4)

We selected db4 for:
- **Smoothness**: Continuous first derivative
- **Compact support**: Good localization in both time and frequency
- **Common choice**: Widely used in image processing

Other options (configurable): `haar`, `sym4`, `bior4.4`

### 5.2 Decomposition Levels: 3

With 3 levels on a 1024×1024 image:

| Level | Resolution | What it captures |
|-------|------------|------------------|
| LL | 128×128 | Global structure, average color |
| Level 1 | 256×256 | Fine texture, noise |
| Level 2 | 512×512 | Medium details |
| Level 3 | 512×512 | Coarse details, major edges |

### 5.3 Boundary Smoothing

To prevent artifacts at mask boundaries, we apply Gaussian smoothing:

```python
def apply_boundary_smoothing(initialized, original, mask, sigma=2.0):
    # Create boundary region using dilation - erosion
    # Apply smooth transition between initialized and original
```

### 5.4 Code Structure

```
wavelet-inpainting/
├── models/
│   ├── wavelet_init.py      # Core wavelet initialization algorithm
│   └── inpainting_pipeline.py # SD integration and full pipeline
├── baselines/
│   ├── pure_noise.py        # Baseline: pure Gaussian noise
│   └── naive_blend.py       # Baseline: uniform α=0.5 blending
├── data/
│   ├── celeba_hq_loader.py  # Dataset loading with train/val/test splits
│   └── mask_generator.py    # Mask generation (center, random, irregular)
├── evaluation/
│   └── metrics.py           # FID, LPIPS, PSNR, SSIM computation
├── visualization/
│   └── plots.py             # Visualization and comparison plots
├── experiments/
│   └── run_celeba_experiment.py # Main experiment script
└── configs/
    └── default_config.yaml  # Experiment configuration
```

---

## 6. Experimental Design

### 6.1 Dataset: CelebA-HQ

- **Total images**: 30,000 high-quality face images
- **Resolution**: 1024×1024 pixels
- **Source**: NVIDIA Progressive GAN dataset

### 6.2 Data Split

```python
# Fixed random seed (42) for reproducibility
n_train = int(0.8 * n_total)  # 24,000 images
n_val = int(0.1 * n_total)    #  3,000 images
n_test = n_total - n_train - n_val  # 3,000 images
```

**Evaluation uses only the TEST split** to ensure unbiased results.

### 6.3 Mask Configuration

- **Type**: Center square mask
- **Size**: 256×256 pixels (covers central 6.25% of image area)
- **Purpose**: Consistent evaluation across all methods

### 6.4 Methods Compared

| Method | Description | Parameters |
|--------|-------------|------------|
| **Pure Noise** | Fill masked region with Gaussian noise | σ = 0.5 |
| **Naive Blend** | Uniform blend: `0.5 × context + 0.5 × noise` | α = 0.5, σ = 0.5 |
| **Wavelet (Ours)** | Frequency-aware blending | α = [0.9, 0.6, 0.3], σ = 0.5 |

### 6.5 Diffusion Settings

| Parameter | Value |
|-----------|-------|
| Model | `runwayml/stable-diffusion-inpainting` |
| Inference steps | 50 |
| Guidance scale | 7.5 |
| Prompt | "" (empty for unconditional) |
| Precision | float16 (GPU) |

---

## 7. Evaluation Metrics

### 7.1 Metrics Overview

| Metric | Measures | Better | Range |
|--------|----------|--------|-------|
| **FID** | Distribution similarity (Inception features) | Lower ↓ | [0, ∞) |
| **LPIPS** | Perceptual similarity (learned features) | Lower ↓ | [0, 1] |
| **PSNR** | Pixel-wise reconstruction error | Higher ↑ | [0, ∞) dB |
| **SSIM** | Structural similarity | Higher ↑ | [0, 1] |

### 7.2 Metric Computation

**PSNR (Peak Signal-to-Noise Ratio)**:
```
PSNR = 20 × log₁₀(MAX / √MSE)
```
Computed on masked region only.

**SSIM (Structural Similarity Index)**:
```
SSIM = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / (μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)
```
Computed with Gaussian weighting, window size 7.

**LPIPS (Learned Perceptual Image Patch Similarity)**:
- Uses AlexNet features
- Measures perceptual difference
- Crops to masked region bounding box

**FID (Fréchet Inception Distance)**:
```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^½)
```
Computed on Inception-v3 features of full images.

### 7.3 Statistical Reporting

For PSNR, SSIM, LPIPS: Report **mean ± std** across test images.
For FID: Report single value (distribution-level metric).

---

## 8. Project Architecture

### 8.1 Class Diagram

```
┌─────────────────────────────────────────┐
│         WaveletInpaintingPipeline       │
├─────────────────────────────────────────┤
│ - sd_pipeline: StableDiffusionInpaint   │
│ - device: str                           │
├─────────────────────────────────────────┤
│ + initialize_with_wavelet()             │
│ + inpaint()                             │
│ + inpaint_batch()                       │
└─────────────────────────────────────────┘
                    │ uses
                    ▼
┌─────────────────────────────────────────┐
│           wavelet_initialize()          │
├─────────────────────────────────────────┤
│ Core algorithm:                         │
│ - DWT decomposition                     │
│ - Frequency-aware blending              │
│ - IDWT reconstruction                   │
└─────────────────────────────────────────┘
                    │ compared against
                    ▼
┌───────────────────┬─────────────────────┐
│ pure_noise_init() │ naive_blend_init()  │
└───────────────────┴─────────────────────┘
```

### 8.2 Data Flow

```
CelebAHQDataset
      │
      │ provides (image, mask)
      ▼
run_experiment()
      │
      ├──▶ Pure Noise Init ──▶ SD Inpaint ──▶ Result₁
      │
      ├──▶ Naive Blend Init ─▶ SD Inpaint ──▶ Result₂
      │
      └──▶ Wavelet Init ────▶ SD Inpaint ──▶ Result₃
                                              │
                                              ▼
                                    MetricsCalculator
                                              │
                                              ▼
                                    Aggregate Results
```

### 8.3 Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `wavelet_initialize()` | `models/wavelet_init.py` | Core algorithm |
| `blend_band()` | `models/wavelet_init.py` | Blend single wavelet band |
| `downsample_mask()` | `models/wavelet_init.py` | Max-pool mask for each level |
| `pure_noise_initialize()` | `baselines/pure_noise.py` | Baseline: pure noise |
| `naive_blend_initialize()` | `baselines/naive_blend.py` | Baseline: uniform blend |
| `compute_all_metrics()` | `evaluation/metrics.py` | PSNR, SSIM, LPIPS |
| `compute_fid()` | `evaluation/metrics.py` | FID computation |
| `run_experiment()` | `experiments/run_celeba_experiment.py` | Main experiment loop |

---

## 9. Reproducibility

### 9.1 Random Seeds

All random operations use fixed seeds for reproducibility:

```python
config['seed'] = 42

# Data split
rng = np.random.RandomState(42)
indices = rng.permutation(n_total)

# Per-image noise
seed = config['seed'] + idx  # Different seed per image, but deterministic
```

### 9.2 Environment

```
Python: 3.12+
PyTorch: 2.0+
Key packages:
  - diffusers >= 0.21.0
  - PyWavelets >= 1.4.1
  - pytorch-fid >= 0.3.0
  - lpips >= 0.1.4
```

### 9.3 Running Experiments

```bash
# Quick validation
python experiments/run_celeba_experiment.py --validate

# Full experiment (500 samples)
python experiments/run_celeba_experiment.py --samples 500

# With synthetic data (no dataset needed)
python experiments/run_celeba_experiment.py --synthetic --mock
```

### 9.4 Expected Outputs

```
outputs/experiment_YYYYMMDD_HHMMSS/
├── config.json              # Experiment configuration
├── summary.txt              # Text summary of results
├── comparisons/             # Side-by-side comparison images
├── initializations/         # Init method visualization
├── wavelet_viz/             # Wavelet decomposition figures
└── metrics/
    ├── metrics.json         # Quantitative results
    └── metrics_table.png    # Visual metrics table
```

---

## 10. Theoretical Justification

### 10.1 Why Wavelets?

Wavelets provide **localized frequency decomposition**:
- Unlike Fourier transform (global), wavelets capture local frequency content
- Multi-resolution analysis naturally separates structure from detail
- Computationally efficient (O(n) for DWT)

### 10.2 Why Trust Low Frequencies?

Low-frequency components encode:
- **Illumination**: Overall brightness, lighting direction
- **Color tone**: Average skin color, ambient color
- **Structure**: Face shape, feature positions

These should be consistent with surrounding context for seamless results.

### 10.3 Why Allow Noise in High Frequencies?

High-frequency components encode:
- **Texture**: Skin pores, hair strands
- **Fine details**: Wrinkles, subtle features
- **Noise**: Sensor noise, compression artifacts

The diffusion model is skilled at generating realistic high-frequency content. Providing too much context here may limit diversity or create artifacts.

### 10.4 Connection to Diffusion Process

Diffusion models work by gradually denoising from pure noise. Our initialization:
- **Guides** low-frequency convergence toward context-consistent results
- **Allows** high-frequency generation freedom for realistic details
- **Reduces** the "work" the diffusion model must do for structure

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

1. **Fixed alpha schedule**: Currently hand-tuned; could be learned
2. **Center masks only**: Irregular masks less tested
3. **Face-specific evaluation**: CelebA-HQ only; needs broader evaluation
4. **Single wavelet type**: db4 only; other wavelets may work better

### 11.2 Future Directions

1. **Learned alpha schedule**: Train to predict optimal α per image/mask
2. **Mask-adaptive blending**: Vary α based on mask shape/location
3. **Other domains**: Extend to scenes, objects, medical imaging
4. **Integration with fine-tuned models**: Test with LoRA/DreamBooth models

---

## 12. Citation

```bibtex
@misc{wavelet-inpainting,
  title={Adaptive Wavelet-Based Context Initialization for Diffusion Inpainting},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/wavelet-inpainting}}
}
```

---

*Document generated from codebase analysis. Last updated: January 2026.*
