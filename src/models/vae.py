import torch
import torch.nn as nn


def _get_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if activation == "gelu":
        return nn.GELU()
    if activation == "elu":
        return nn.ELU()
    if activation == "silu":
        return nn.SiLU()

    raise ValueError(
        "Unsupported activation. Choose from: "
        "relu, leaky_relu, gelu, elu, silu"
    )


#  Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = _get_activation(activation)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


#  Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim,
                 kernel_size, stride, padding, use_residual, input_size,
                 activation="relu"):
        super().__init__()

        layers = []
        c = in_channels

        for h in hidden_dims:
            layers.append(nn.Conv2d(c, h, kernel_size, stride, padding))
            layers.append(_get_activation(activation))

            if use_residual:
                layers.append(ResBlock(h, activation=activation))

            c = h

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        #  auto compute shape
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.conv(dummy)
            self.feature_shape = out.shape[1:]
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)


#  Decoder
class Decoder(nn.Module):
    def __init__(self, feature_shape, hidden_dims, latent_dim,
                 kernel_size, stride, padding, use_residual,
                 activation="relu"):
        super().__init__()

        hidden_dims = hidden_dims[::-1]
        self.feature_shape = feature_shape

        self.fc = nn.Linear(latent_dim, int(torch.prod(torch.tensor(feature_shape))))

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                   kernel_size, stride, padding)
            )
            layers.append(_get_activation(activation))

            if use_residual:
                layers.append(ResBlock(hidden_dims[i+1], activation=activation))

        self.deconv = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], 3,
                               kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self.feature_shape)
        x = self.deconv(x)
        return self.final(x)


#  VAE
class VAE(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        hidden_dims=[32, 64, 128],
        kernel_size=4,
        stride=2,
        padding=1,
        use_residual=False,
        input_size=64,
        activation="relu",
    ):
        super().__init__()

        self.encoder = Encoder(
            3, hidden_dims, latent_dim,
            kernel_size, stride, padding,
            use_residual, input_size,
            activation=activation,
        )

        self.decoder = Decoder(
            self.encoder.feature_shape,
            hidden_dims, latent_dim,
            kernel_size, stride, padding,
            use_residual,
            activation=activation,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar