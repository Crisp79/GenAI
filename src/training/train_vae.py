import torch
from tqdm import tqdm
from src.training.vae_loss import vae_loss


def train_vae(model, dataloader, optimizer, device,beta):
    model.train()

    total_loss = 0

    for batch in tqdm(dataloader):
        x, _ = batch
        x = x.to(device)

        optimizer.zero_grad()

        recon, mu, logvar = model(x)

        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
