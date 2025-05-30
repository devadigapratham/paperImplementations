import torch 
from torch import nn 
from torch.nn import functional as F
import math 
import os 
import shutil 
from sklearn.model_selection import train_test_split 
import os 

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embd_dim, in_proj_bias = True, out_proj_bias = True): 
        super().__init__() 
        self.n_heads = n_heads 
        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
        self.d_heads = embd_dim // n_heads 

    def forward(self, x, casual_mask = False):
        batch_size, seq_len, d_emed = x.shape 
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1) 

        output = weight @ v 
        output = output.transpose(1, 2)
        output = output.reshape((batch_size, seq_len, d_emed))
        output = self.out_proj(output)

        return output 
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__() 
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residual = x.clone() 
        x = self.groupnorm(x) 

        n, c, h, w = x.shape 
        
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residual

        return x  

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity() 
        else:
            self.residual_layer = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x.clone() 

        x = self.groupnorm1(x) 
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x) 

        return x + self.residual_layer(residue)

class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding = 1), 
            ResidualBlock(128, 128), 
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(128, 256),
            ResidualBlock(256, 256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(256, 512), 
            ResidualBlock(512, 512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            ResidualBlock(512, 512), 
            nn.GroupNorm(32, 512),  
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0) 
        )

        def forward(self, x):
            for module in self:
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    x = F.pad(x, (0, 1, 0, 1))
                x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std) 
        x = mean + eps * std 

        x *= 0.19215 # scaling the latent representation 
        return x 

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1), 
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            ResidualBlock(256, 128), 
            ResidualBlock(128, 128), 
            ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215 #remove the scaling added by the encoder 

        for module in self:
            x = module(x)
        
        return x # (batch_size, 3, h, w) 
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder() 
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded 
        
    
