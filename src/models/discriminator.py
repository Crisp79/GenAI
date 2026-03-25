# src/models/discriminator.py

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        channels=[64, 128, 256, 512],
        in_channels=3,
        use_batchnorm=True
    ):
        super().__init__()

        layers = []

        # First layer (no BN)
        layers.append(nn.Conv2d(in_channels, channels[0], 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_ch = channels[0]

        for ch in channels[1:]:
            layers.append(nn.Conv2d(in_ch, ch, 4, 2, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch * 4 * 4, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)