import torch
import torch.nn as nn


# 🔷 Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x)


# 🔷 Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[32, 64, 128, 256],
        latent_dim=128,
        kernel_size=4,
        stride=2,
        padding=1,
        use_residual=False,
        input_size=64,
    ):
        super().__init__()

        modules = []
        current_channels = in_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, h_dim, kernel_size, stride, padding),
                    nn.ReLU(),
                )
            )

            if use_residual:
                modules.append(ResBlock(h_dim))

            current_channels = h_dim

        self.conv = nn.Sequential(*modules)
        self.flatten = nn.Flatten()

        # 🔥 Dynamically compute feature map size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.conv(dummy)
            self.feature_shape = out.shape[1:]  # (C, H, W)
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# 🔷 Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        feature_shape,
        out_channels=3,
        hidden_dims=[32, 64, 128, 256],
        latent_dim=128,
        kernel_size=4,
        stride=2,
        padding=1,
        use_residual=False,
    ):
        super().__init__()

        self.feature_shape = feature_shape
        hidden_dims = hidden_dims[::-1]

        self.fc = nn.Linear(latent_dim, int(torch.prod(torch.tensor(feature_shape))))

        modules = []
        current_channels = hidden_dims[0]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size,
                        stride,
                        padding,
                    ),
                    nn.ReLU(),
                )
            )

            if use_residual:
                modules.append(ResBlock(hidden_dims[i + 1]))

        self.deconv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self.feature_shape)
        x = self.deconv(x)
        x = self.final_layer(x)
        return x


# 🔷 VAE Wrapper
class VAE(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
        kernel_size=4,
        stride=2,
        padding=1,
        use_residual=False,
        input_size=64,
    ):
        super().__init__()

        # Encoder
        self.encoder = Encoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_residual=use_residual,
            input_size=input_size,
        )

        # Decoder (uses encoder's computed shape)
        self.decoder = Decoder(
            feature_shape=self.encoder.feature_shape,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_residual=use_residual,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar