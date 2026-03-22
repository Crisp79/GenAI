import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDownBlock(nn.Module):
    """Encoder Block: Halves spatial dimensions and applies a residual shortcut."""
    def __init__(self, in_channels, out_channels, kernel_size=3, act_fn=nn.ReLU, bn_first=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # Dynamically build the main path based on the ablation toggle
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding))
        
        if bn_first:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn())
        else:
            layers.append(act_fn())
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding))
        layers.append(nn.BatchNorm2d(out_channels)) # Always normalize before the final residual addition
        
        self.main = nn.Sequential(*layers)
        self.channel_match = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = act_fn()

    def forward(self, x):
        out = self.main(x)
        x_shortcut = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_shortcut = self.channel_match(x_shortcut)
        return self.act_fn(out + x_shortcut)

class ResUpBlock(nn.Module):
    """Decoder Block: Doubles spatial dimensions and applies a residual shortcut."""
    def __init__(self, in_channels, out_channels, kernel_size=4, act_fn=nn.ReLU, bn_first=True):
        super().__init__()
        
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1))
        
        if bn_first:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn())
        else:
            layers.append(act_fn())
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.main = nn.Sequential(*layers)
        self.channel_match = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = act_fn()

    def forward(self, x):
        out = self.main(x)
        x_shortcut = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x_shortcut = self.channel_match(x_shortcut)
        return self.act_fn(out + x_shortcut)

class VAE(nn.Module):
    # Added the bn_first toggle to the main model initialization
    def __init__(self, latent_dim=128, kernel_size=3, init_channels=32, act_fn=nn.ReLU, bn_first=True):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(init_channels) if bn_first else nn.Identity(),
            act_fn(),
            nn.BatchNorm2d(init_channels) if not bn_first else nn.Identity(),
            
            # Pass the toggle and kernel size down to the blocks
            ResDownBlock(init_channels, init_channels * 2, kernel_size, act_fn, bn_first),     
            ResDownBlock(init_channels * 2, init_channels * 4, kernel_size, act_fn, bn_first), 
            ResDownBlock(init_channels * 4, init_channels * 8, kernel_size, act_fn, bn_first), 
            nn.Flatten()
        )
        
        flat_size = (init_channels * 8) * 8 * 8
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (init_channels * 8, 8, 8)),
            ResUpBlock(init_channels * 8, init_channels * 4, kernel_size=4, act_fn=act_fn, bn_first=bn_first), 
            ResUpBlock(init_channels * 4, init_channels * 2, kernel_size=4, act_fn=act_fn, bn_first=bn_first), 
            ResUpBlock(init_channels * 2, init_channels, kernel_size=4, act_fn=act_fn, bn_first=bn_first),     
            
            nn.Conv2d(init_channels, 3, kernel_size=3, stride=1, padding=1),
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