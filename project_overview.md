# Face VAE Project — Complete Technical Overview

## 1. Project Objective

This project implements a Variational Autoencoder (VAE) to model and generate face images with and without glasses. The pipeline is designed to be modular, reproducible, and suitable for ablation studies.

The system performs:

* Data preprocessing and augmentation
* Model training (VAE)
* Ablation experiments
* Evaluation using reconstruction quality and SSIM
* Image generation from latent space

---

## 2. Project Structure

```
face-vae-project/
│
├── data/
│   ├── raw/                           # not in github
│   ├── processed_64/                  # resized images (64x64)
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── audit/clip_audit.py
│   │
│   ├── models/
│   │   └── vae.py
│   │
│   ├── training/
│   │   ├── train_vae.py
│   │   └── vae_loss.py
│   │
│   ├── evaluation/
│   │   ├── vae_visualize.py
│   │   └── vae_generate.py
│   │
│   └── experiments/
│       └── run_vae_ablation.py
│
├── notebooks/
│   └── 03_experiments.ipynb
│
├── outputs/
│   └── (results saved with ../ prefix)
│
└── main.py
```

---

## 3. Data Pipeline

### Dataset

* Images are stored in `data/processed/` as 64x64 RGB images.
* Labels are stored in CSV files.
* Label:

  * 0 → no glasses
  * 1 → glasses

### Dataset Class (`FacesDataset`)

* Loads images using OpenCV
* Converts BGR to RGB
* Applies Albumentations transforms
* Returns `(image, label)`

### DataLoader

* Batch size: ~32
* Train loader is shuffled
* Output shape: `(B, 3, 64, 64)`

---

## 4. Model: Variational Autoencoder

### Key Components

#### Encoder

* Stack of convolutional layers
* Downsamples image spatially
* Outputs:

  * Mean (`mu`)
  * Log variance (`logvar`)

#### Latent Sampling

Uses reparameterization trick:

```
z = mu + std * epsilon
```

#### Decoder

* Transposed convolutions
* Reconstructs image from latent vector

#### Residual Blocks

Optional:

```
x = x + f(x)
```

Improves stability and gradient flow.

---

## 5. Dynamic Architecture Design

The model is fully configurable:

* `latent_dim`
* `hidden_dims`
* `kernel_size`
* `stride`
* `padding`
* `use_residual`

### Important Detail

The encoder dynamically computes output feature size using a dummy forward pass. This avoids hardcoding dimensions.

---

## 6. Loss Function

Defined in `vae_loss.py`:

```
Loss = Reconstruction Loss + beta * KL Divergence
```

### Reconstruction Loss

* L1 loss (preferred for sharper images)

### KL Divergence

* Regularizes latent space

### Beta

* Controls trade-off
* Annealed during training

---

## 7. Training Pipeline

### Training Loop (`train_vae.py`)

For each batch:

1. Forward pass
2. Compute loss
3. Backpropagation
4. Optimizer step

### KL Annealing

```
beta increases from 0 → 0.05 over epochs
```

This prevents posterior collapse.

---

## 8. Ablation Experiments

Defined in `run_vae_ablation.py`

Each experiment:

* Runs for 1 epoch (as required)
* Logs:

  * Loss
  * SSIM
* Saves images

### Example Variations

* Latent dimension
* Network depth
* Residual connections

---

## 9. Evaluation

### SSIM (Structural Similarity)

Measures perceptual similarity between:

* Original image
* Reconstruction

Before computing SSIM:

* Reconstructions are resized to match input size

### Visualization (`vae_visualize.py`)

Saves a 3-row grid:

1. Original images
2. Reconstructions
3. Generated samples

Saved to:

```
../outputs/vae_images/
```

---

## 10. Image Generation

Defined in `vae_generate.py`

### Process

1. Sample latent vector:

```
z ~ N(0,1)
```

2. Pass through decoder
3. Generate images

### Output

* Generates 12 sampled images (final configuration)
* Saved to:

```
../outputs/final_vae_samples.png
```

---

## 11. Final Model Selection

Model is selected based on:

* Highest SSIM
* Best visual reconstruction
* Stability

Residual model was chosen due to:

* Better structure preservation
* Improved reconstruction quality

---

## 12. Training Trends

During full training:

* Loss decreases over epochs
* SSIM increases and stabilizes

These trends indicate:

* Improved reconstruction
* Better latent representation

---

## 13. Key Design Decisions

* 64x64 resolution for efficiency
* Modular structure (separation of concerns)
* Dynamic model architecture
* Notebook used only for orchestration
* All logic inside `src/`

---

## 14. Important Notes

* Output paths use `../outputs/`
* Final generated images include only 12 samples
* Kernel size variations were avoided due to shape mismatch issues
* Reconstruction outputs are resized when needed for evaluation consistency

---

## 15. Limitations

* VAE produces blurry images due to pixel-wise loss
* No control over attributes (glasses vs no glasses)
* Latent space is not explicitly disentangled

---

## 16. Possible Extensions

* Conditional VAE (for attribute control)
* GAN implementation for sharper images
* Diffusion models for higher quality generation
* FID score for evaluation

---

## 17. Summary

This project implements a complete generative pipeline using a VAE with:

* Clean architecture
* Flexible configuration
* Reproducible experiments
* Proper evaluation and visualization

It serves as a strong foundation for extending into more advanced generative models.
