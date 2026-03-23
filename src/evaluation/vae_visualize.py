import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F

def compute_ssim(x, recon):
    # 🔥 Ensure same size
    if recon.shape != x.shape:
        recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear")

    x = x.cpu().numpy()
    recon = recon.cpu().numpy()

    scores = []

    for i in range(len(x)):
        img1 = np.transpose(x[i], (1,2,0))
        img2 = np.transpose(recon[i], (1,2,0))

        score = ssim(img1, img2, channel_axis=2, data_range=1.0)
        scores.append(score)

    return np.mean(scores)


def save_images(original, recon, samples, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)

    n = 6

    original = original[:n].cpu().permute(0,2,3,1)
    recon = recon[:n].cpu().permute(0,2,3,1)
    samples = samples[:n].cpu().permute(0,2,3,1)

    fig, axes = plt.subplots(1, n, figsize=(12,6))

    for i in range(n):
        axes[i].imshow(samples[i])
        axes[i].axis("off")

    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()


def evaluate_vae(model, train_loader, device, latent_dim, save_dir, name):
    model.eval()

    x, _ = next(iter(train_loader))
    x = x.to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

        z = torch.randn(x.size(0), latent_dim).to(device)
        samples = model.decoder(z)

    ssim_score = compute_ssim(x, recon)

    save_images(x, recon, samples, save_dir, name)

    return ssim_score