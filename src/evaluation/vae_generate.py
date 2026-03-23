import torch
import os
import matplotlib.pyplot as plt


def generate_vae_samples(
    model,
    num_samples,
    latent_dim,
    device,
    save_path=None,
    show=True
):
    model.eval()

    with torch.no_grad():
        # 🔷 Sample from latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z)

    # 🔷 Move to CPU for visualization
    samples_cpu = samples.cpu()

    # 🔷 Plot
    if show:
        n = num_samples
        cols = min(n, 6)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            img = samples_cpu[i].permute(1,2,0)
            axes[i].imshow(img)
            axes[i].axis("off")

        # Hide extra axes
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        plt.show()

    return samples_cpu