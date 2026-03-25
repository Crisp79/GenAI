import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.notebook import tqdm

from src.models.vae import VAE


def _to_metric_uint8(images):
    images = torch.clamp(images.detach(), 0.0, 1.0)
    return (images * 255.0).to(torch.uint8)


def _safe_fid_is(real_images, fake_images, device, fid_feature=64, is_splits=5):
    real_uint8 = _to_metric_uint8(real_images)
    fake_uint8 = _to_metric_uint8(fake_images)

    fid_score = float("nan")
    is_mean = float("nan")

    try:
        fid_metric = FrechetInceptionDistance(feature=fid_feature).to(device)
        fid_metric.update(real_uint8.to(device), real=True)
        fid_metric.update(fake_uint8.to(device), real=False)
        fid_score = float(fid_metric.compute().detach().cpu())
    except Exception:
        pass

    try:
        is_metric = InceptionScore(splits=is_splits).to(device)
        is_metric.update(fake_uint8.to(device))
        is_mean_tensor, _ = is_metric.compute()
        is_mean = float(is_mean_tensor.detach().cpu())
    except Exception:
        pass

    return fid_score, is_mean


def _write_checkpoint_manifest(records, save_dir, model_prefix):
    if not records or not save_dir:
        return None

    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, f"{model_prefix}_checkpoint_manifest.csv")
    fieldnames = [
        "epoch",
        "loss",
        "fid",
        "is",
        "beta",
        "model_path",
        "image_path",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return manifest_path


def save_images(original, recon, samples, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)

    n = 6
    samples = samples[:n].cpu().permute(0, 2, 3, 1)

    # Use a near-square layout so saved ablation images are not linear strips.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    if cols < rows:
        cols, rows = rows, cols
        rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        axes[i].imshow(samples[i])
        axes[i].axis("off")

    for i in range(n, len(axes)):
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

    fid_score, is_mean = _safe_fid_is(x, samples, device)
    save_images(x, recon, samples, save_dir, name)
    return {
        "fid": fid_score,
        "is": is_mean,
    }


def vae_loss(
    recon_x,
    x,
    mu,
    logvar,
    beta=0.01,
    recon_loss_type="l1",
    free_bits=0.0,
):
    # Keep this resize behavior to preserve current training behavior.
    if recon_x.shape != x.shape:
        recon_x = F.interpolate(recon_x, size=x.shape[-2:], mode="bilinear")

    if recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    elif recon_loss_type == "smooth_l1":
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction="mean")
    else:
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")

    # Clamp logvar to avoid overflow in exp(logvar).
    logvar = torch.clamp(logvar, min=-20.0, max=20.0)

    if beta == 0:
        kl_loss = torch.zeros((), device=x.device, dtype=recon_loss.dtype)
        total_loss = recon_loss
    else:
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits and free_bits > 0:
            kl_loss = torch.clamp(kl_loss, min=free_bits)
        kl_loss = kl_loss.mean()
        total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae_full(
    train_loader,
    device,
    # Model config
    latent_dim=64,
    hidden_dims=[32, 64, 128],
    kernel_size=4,
    stride=2,
    padding=1,
    use_residual=True,
    activation="relu",
    # Training config
    lr=1e-3,
    num_epochs=20,
    beta_start=0.0,
    beta_end=0.05,
    beta_warmup_epochs=None,
    recon_loss_type="l1",
    free_bits=0.0,
    grad_clip_norm=None,
    metric_interval=1,
    checkpoint_interval=None,
    model_save_dir=None,
    image_save_dir=None,
    model_prefix="vae",
):
    model = VAE(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        use_residual=use_residual,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    fid_history = []
    is_history = []
    checkpoint_records = []

    if beta_warmup_epochs is None:
        beta_warmup_epochs = num_epochs
    beta_warmup_epochs = max(1, int(beta_warmup_epochs))

    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        model.train()
        total_loss = 0

        beta_progress = min(epoch_num, beta_warmup_epochs) / beta_warmup_epochs
        beta = beta_start + (beta_end - beta_start) * beta_progress

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch_num}/{num_epochs}"):
            x = x.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon,
                x,
                mu,
                logvar,
                beta=beta,
                recon_loss_type=recon_loss_type,
                free_bits=free_bits,
            )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        x, _ = next(iter(train_loader))
        x = x.to(device)

        with torch.no_grad():
            recon, _, _ = model(x)
            z = torch.randn(x.size(0), latent_dim).to(device)
            samples = model.decoder(z)

        should_eval_metrics = (
            metric_interval is None
            or metric_interval <= 1
            or epoch_num % metric_interval == 0
            or epoch_num == num_epochs
        )

        if should_eval_metrics:
            fid_score, is_mean = _safe_fid_is(x, samples, device)
        else:
            fid_score, is_mean = np.nan, np.nan

        loss_history.append(avg_loss)
        fid_history.append(fid_score)
        is_history.append(is_mean)

        if (
            checkpoint_interval is not None
            and checkpoint_interval > 0
            and epoch_num % checkpoint_interval == 0
        ):
            model_path = None
            image_path = None

            if model_save_dir:
                os.makedirs(model_save_dir, exist_ok=True)
                model_path = os.path.join(
                    model_save_dir,
                    f"{model_prefix}_epoch_{epoch_num}.pth",
                )
                torch.save(model.state_dict(), model_path)

            if image_save_dir:
                os.makedirs(image_save_dir, exist_ok=True)
                image_path = os.path.join(
                    image_save_dir,
                    f"{model_prefix}_epoch_{epoch_num}.png",
                )
                save_images(
                    x,
                    recon,
                    samples,
                    image_save_dir,
                    f"{model_prefix}_epoch_{epoch_num}",
                )

            checkpoint_records.append(
                {
                    "epoch": epoch_num,
                    "loss": float(avg_loss),
                    "fid": float(fid_score),
                    "is": float(is_mean),
                    "beta": float(beta),
                    "model_path": model_path or "",
                    "image_path": image_path or "",
                }
            )

        print(
            f"Epoch {epoch_num}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"FID: {fid_score:.4f} | "
            f"IS: {is_mean:.4f} | "
            f"Beta: {beta:.4f}"
        )

    manifest_target_dir = model_save_dir or image_save_dir
    manifest_path = _write_checkpoint_manifest(
        checkpoint_records,
        save_dir=manifest_target_dir,
        model_prefix=model_prefix,
    )
    if manifest_path:
        print(f"Checkpoint manifest saved to: {manifest_path}")

    return model, loss_history, fid_history, is_history


def generate_vae_samples(
    model,
    num_samples,
    latent_dim,
    device,
    save_path=None,
    show=True,
    temperature=1.0,
):
    model.eval()

    with torch.no_grad():
        temperature = max(float(temperature), 1e-6)
        z = torch.randn(num_samples, latent_dim).to(device) * temperature
        samples = model.decoder(z)

    samples_cpu = samples.cpu()

    if show:
        n = num_samples
        # Prefer a near-square grid while keeping it horizontal.
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        if cols < rows:
            cols, rows = rows, cols
            rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            img = samples_cpu[i].permute(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        plt.show()

    return samples_cpu
