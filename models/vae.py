import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class VAE(nn.Module):
    def __init__(self,latent_dim = 128,kernel_size=3,init_channels = 32,act_fn=nn.ReLU):
        super(VAE,self).__init__()
        padding = (kernel_size - 1) // 2

        # initial size : 64x64x3
        self.encoder = nn.Sequential(
            nn.Conv2d(3,init_channels,kernel_size = kernel_size,stride=2,padding=padding),
            act_fn(),
            nn.Conv2d(init_channels,init_channels*2,kernel_size = kernel_size,stride=2,padding=padding),
            act_fn(),
            nn.Conv2d(init_channels*2,init_channels*4,kernel_size = kernel_size,stride=2,padding=padding),
            act_fn(),
            nn.Flatten()
        )

        flat_size = (init_channels * 4) * 8 * 8
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (init_channels * 4, 8, 8)),
            nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.ConvTranspose2d(init_channels * 2, init_channels, kernel_size=4, stride=2, padding=1),
            act_fn(),
            nn.ConvTranspose2d(init_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.fc_decode(z)), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD