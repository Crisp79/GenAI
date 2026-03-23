import torch
from src.training.loss import vae_loss

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            total_loss += loss.item()

    return total_loss