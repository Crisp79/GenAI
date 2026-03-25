from tqdm.notebook import tqdm
import torch.nn as nn
import torch
from src.models.generator import Generator
from src.models.discriminator import Discriminator

def train_gan(
    generator,
    discriminator,
    dataloader,
    device,
    epochs=20,
    latent_dim=128,
    lr=2e-4,
    checkpoint_callback=None
):
    

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    for epoch in range(epochs):

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for real_imgs, _ in loop:

            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # optional label noise
            real_labels += 0.05 * torch.rand_like(real_labels)
            fake_labels += 0.05 * torch.rand_like(fake_labels)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            loss_real = criterion(discriminator(real_imgs), real_labels)
            loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
            loss_D = loss_real + loss_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            loss_G = criterion(discriminator(fake_imgs), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            #  Update progress bar
            loop.set_postfix({
                "D_loss": f"{loss_D.item():.3f}",
                "G_loss": f"{loss_G.item():.3f}"
            })

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        
        # Call checkpoint callback if provided
        if checkpoint_callback is not None:
            checkpoint_callback(epoch)

    return G_losses, D_losses

def train_gan_full(config,device,train_loader,epochs):
    G = Generator(
        latent_dim=config["latent_dim"],
        channels=config["g_channels"],
        use_batchnorm=config["use_batchnorm"],
        activation=config["activation"]
    ).to(device)

    D = Discriminator(
        channels=config["d_channels"],
        use_batchnorm=config["use_batchnorm"]
    ).to(device)

    G_losses, D_losses = train_gan(
        generator=G,
        discriminator=D,
        dataloader=train_loader,
        device=device,
        epochs=epochs,
        latent_dim=config["latent_dim"],
        lr=config["lr"]
    )

    return G,D,G_losses,D_losses


def build_gan_models(config, device):
    """Create generator/discriminator pair from config."""
    generator = Generator(
        latent_dim=config["latent_dim"],
        channels=config["g_channels"],
        use_batchnorm=config["use_batchnorm"],
        activation=config["activation"],
    ).to(device)

    discriminator = Discriminator(
        channels=config["d_channels"],
        use_batchnorm=config["use_batchnorm"],
    ).to(device)

    return generator, discriminator


def train_gan_with_epoch_callback(
    generator,
    discriminator,
    dataloader,
    device,
    epochs=20,
    latent_dim=128,
    lr=2e-4,
    epoch_callback=None,
):
    """Train GAN and invoke callback after each epoch with latest losses."""
    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            real_labels += 0.05 * torch.rand_like(real_labels)
            fake_labels += 0.05 * torch.rand_like(fake_labels)

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            loss_real = criterion(discriminator(real_imgs), real_labels)
            loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
            loss_D = loss_real + loss_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)

            loss_G = criterion(discriminator(fake_imgs), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({
                "D_loss": f"{loss_D.item():.3f}",
                "G_loss": f"{loss_G.item():.3f}",
            })

        g_loss_epoch = float(loss_G.item())
        d_loss_epoch = float(loss_D.item())
        G_losses.append(g_loss_epoch)
        D_losses.append(d_loss_epoch)

        if epoch_callback is not None:
            epoch_callback(
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                g_loss=g_loss_epoch,
                d_loss=d_loss_epoch,
            )

    return G_losses, D_losses