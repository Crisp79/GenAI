import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, beta=0.01):

    # 🔥 Fix shape mismatch
    if recon_x.shape != x.shape:
        recon_x = F.interpolate(recon_x, size=x.shape[-2:], mode="bilinear")

    recon_loss = F.l1_loss(recon_x, x, reduction="mean")

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss.mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss