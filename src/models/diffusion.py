import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision


# Time Embedding

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        device   = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.proj(emb)


# Residual Block

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_c),
        )
        self.conv1 = nn.Conv2d(in_c,  out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act   = nn.SiLU()
        self.skip  = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h     = self.act(self.norm1(self.conv1(x)))
        t_emb = self.time_mlp(t)[:, :, None, None]
        h     = h + t_emb
        h     = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H*W).unbind(1)
        # Standard Scaled Dot-Product Attention
        attn = (q.transpose(-1, -2) @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_mlp = TimeEmbedding(time_dim)

        # Encoder
        self.d1 = ResBlock(3, 64, time_dim)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) # Learned Downsample
        
        self.d2 = ResBlock(64, 128, time_dim)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # Learned Downsample
        
        self.d3 = ResBlock(128, 256, time_dim)
        self.attn3 = Attention(256) # Help with facial symmetry

        # Bottleneck
        self.mid = ResBlock(256, 256, time_dim)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u1 = ResBlock(256 + 128, 128, time_dim)
        self.u2 = ResBlock(128 + 64, 64, time_dim)

        # Output
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Encode
        s1 = self.d1(x, t)
        s2 = self.d2(self.down1(s1), t)
        s3 = self.d3(self.down2(s2), t)
        s3 = self.attn3(s3) # Global context for specs/eyes

        # Bottleneck
        m = self.mid(s3, t)

        # Decode
        u = self.u1(torch.cat([self.up(m), s2], dim=1), t)
        u = self.u2(torch.cat([self.up(u), s1], dim=1), t)

        return self.out_conv(F.silu(self.out_norm(u)))

# Diffusion Model with EMA and Configurable Noise Schedule

class DiffusionModel:
    def __init__(self, device="cuda", timesteps=1000, img_size=64, beta_start=1e-4, beta_end=0.02):
        self.device   = device
        self.T        = timesteps
        self.img_size = img_size

        # Noise schedule (beta_start and beta_end are now configurable)
        self.beta      = torch.linspace(beta_start, beta_end, self.T).to(device)
        self.alpha     = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Model + EMA
        self.model     = UNet().to(device)
        self.ema_model = copy.deepcopy(self.model).eval()

    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_t   = torch.sqrt(a_hat) * x + torch.sqrt(1 - a_hat) * noise
        return x_t, noise

    def update_ema(self, decay=0.995):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(),
                                self.model.parameters()):
                ema_p.data = decay * ema_p.data + (1 - decay) * p.data

    def fit(self, dataloader, epochs=100, lr=1e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                t          = torch.randint(0, self.T, (x.size(0),), device=self.device)
                x_t, noise = self.forward_diffusion(x, t)
                pred_noise = self.model(x_t, t.float())

                loss = F.mse_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.update_ema()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, n=16, use_ema=True):
        model = self.ema_model if use_ema else self.model
        model.eval()

        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

        for t in tqdm(reversed(range(self.T)), total=self.T, desc="Sampling"):
            t_tensor   = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_tensor.float())

            alpha     = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta      = self.beta[t]

            if t > 0:
                alpha_hat_prev = self.alpha_hat[t - 1]
                beta_tilde     = beta * (1 - alpha_hat_prev) / (1 - alpha_hat)
                noise          = torch.randn_like(x)
            else:
                beta_tilde = torch.zeros(1, device=self.device)
                noise      = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
            ) + torch.sqrt(beta_tilde) * noise

        return x   # [-1, 1]