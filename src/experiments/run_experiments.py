import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.models.vae import VAE
from src.training.train_vae import train_vae
from src.evaluation.evaluate import evaluate
from src.data.dataset import FacesDataset
from src.data.transforms import get_train_transform, get_test_transform


def run_experiments(config, train_df, test_df):

    train_loader = DataLoader(
        FacesDataset(train_df, get_train_transform(config["image_size"])),
        batch_size=config["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        FacesDataset(test_df, get_test_transform(config["image_size"])),
        batch_size=config["batch_size"]
    )

    results = []

    for k in config["kernel_sizes"]:
        model = VAE(config["latent_dim"], k).to(config["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        for _ in range(config["epochs"]):
            train_vae(model, train_loader, optimizer, config["device"])

        test_loss = evaluate(model, test_loader, config["device"])

        results.append({
            "kernel_size": k,
            "test_loss": test_loss
        })

    return pd.DataFrame(results)