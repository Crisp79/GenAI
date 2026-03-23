import torch
import pandas as pd
from tqdm import tqdm

from src.models.vae import VAE
from src.training.train_vae import train_vae


def run_vae_ablation(train_loader, device):

    experiments = [
        # (name, config)
        ("baseline", dict(latent_dim=64, hidden_dims=[32,64,128], kernel_size=4, use_residual=False)),
        ("latent_128", dict(latent_dim=128, hidden_dims=[32,64,128], kernel_size=4, use_residual=False)),
        ("deep", dict(latent_dim=64, hidden_dims=[32,64,128,256], kernel_size=4, use_residual=False)),
        ("residual", dict(latent_dim=64, hidden_dims=[32,64,128], kernel_size=4, use_residual=True)),
        ("kernel_3", dict(latent_dim=64, hidden_dims=[32,64,128], kernel_size=3, use_residual=False)),
    ]

    results = []

    for name, config in experiments:
        print(f"\n🚀 Running: {name}")

        model = VAE(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 🔥 1 epoch ONLY (assignment requirement)
        loss = train_vae(
            model,
            train_loader,
            optimizer,
            device,
            beta=0.01
        )

        results.append({
            "experiment": name,
            "latent_dim": config["latent_dim"],
            "depth": len(config["hidden_dims"]),
            "kernel_size": config["kernel_size"],
            "residual": config["use_residual"],
            "loss": loss,
        })

    df = pd.DataFrame(results)
    return df