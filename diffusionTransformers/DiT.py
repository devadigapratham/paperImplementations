import modal 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os 
from dataclasses import dataclass

stub = modal.Stub("DiT")
#Create a Modal image with the necessary dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.0.0",
    "torchvision",
    "numpy",
    "pillow",
    "tqdm",
    "wandb",
    "einops",
    "timm"
])

volume = modal.Volume.from_name("dit-models", create_if_missing=True)

@dataclass 
class DiTConfig:
    img_size: int = 256
    patch_size: int = 2
    in_channels: int = 4 # this is for the latent space (VAE Encoded) 
    hidden_size: int = 1152 
    depth: int = 28 
    num_heads: int = 16
    mlp_ratio: float = 4.0 
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True

def modulate(x, shift, scale): 
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    #Embeddings scalar timesteps into vector representations 

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # Creating sinusoidal embeddings for timesteps
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None] 
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class LabelEmbedder(nn.Module): 
    # Embeds class labels into vector representations

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__() 
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes 
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids = None): 
        if force_drop_ids is None: 
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0 
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
class DiTBlock(nn.Module):
    # DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    def __init__(self, hidden_size, num_heads, mlp_ratio=0.4):
        super().__init_() 
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) 
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout = 0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True), 
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True), 
        ) 

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_output, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        x_norm = self.norm2(x)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels

        # Patch embedding
        self.x_embedder = nn.Linear(config.patch_size ** 2 * config.in_channels, config.hidden_size, bias=True)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_size), requires_grad=False)
        
        # Timestep and label embedders
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size, config.class_dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding
        pos_embed = self.get_2d_sincos_pos_embed(self.config.hidden_size, int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize patch embedding
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def patchify(self, imgs):
        p = self.config.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.config.in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.config.in_channels))
        return x

    def unpatchify(self, x):
        c = self.out_channels
        p = self.config.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        x = self.patchify(x)  # (N, T, D)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
    
class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

@stub.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={"/models": volume},
    timeout=3600,
    secret=modal.Secret.from_name("wandb-secret") 
)
def train_dit():
    import wandb
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    wandb.init(project="dit-diffusion", config={
        "img_size": 256,
        "patch_size": 2,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "num_timesteps": 1000
    })
    
    config = wandb.config
    
    # Initialize model and scheduler
    dit_config = DiTConfig(
        img_size=config.img_size,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads
    )
    
    model = DiT(dit_config).to(device)
    scheduler = DiffusionScheduler(num_timesteps=config.num_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    batch_size = config.batch_size
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        # Dummy training loop
        for step in range(100): 
            # Generate dummy data (replace with real images and labels)
            x = torch.randn(batch_size, 4, 32, 32).to(device)  # Latent space images
            y = torch.randint(0, 1000, (batch_size,)).to(device)  # Class labels
            
            # Sample random timesteps
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,)).to(device)
            
            # Generate random noise
            noise = torch.randn_like(x)
            
            # Add noise to images
            x_noisy = scheduler.q_sample(x, t, noise)
            
            # Predict noise
            predicted_noise = model(x_noisy, t, y)
            
            # Calculate loss
            if dit_config.learn_sigma:
                # Split prediction into noise and variance
                predicted_noise, predicted_var = predicted_noise.chunk(2, dim=1)
            
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(), "step": epoch * 100 + step})
        
        avg_loss = total_loss / 100
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': dit_config
            }
            torch.save(checkpoint, f"/models/dit_checkpoint_epoch_{epoch}.pt")
    
    print("Training completed!")

@stub.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={"/models": volume}
)
def sample_from_dit(checkpoint_path: str, num_samples: int = 4, num_classes: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(f"/models/{checkpoint_path}", map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = DiT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler()
    
    # Generate samples
    with torch.no_grad():
        # Start with random noise
        x = torch.randn(num_samples, config.in_channels, 32, 32).to(device)
        
        # Random class labels
        y = torch.randint(0, num_classes, (num_samples,)).to(device)
        
        # Reverse diffusion process
        for t in reversed(range(scheduler.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long).to(device)
            
            # Predict noise
            predicted_noise = model(x, t_tensor, y)
            
            if config.learn_sigma:
                predicted_noise, _ = predicted_noise.chunk(2, dim=1)
            
            # Remove noise
            alpha_t = scheduler.alphas[t]
            alpha_cumprod_t = scheduler.alphas_cumprod[t]
            beta_t = scheduler.betas[t]
            
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(scheduler.posterior_variance[t]) * noise
    
    return x.cpu()

@stub.local_entrypoint()
def main():
    print("Starting DiT training on Modal...")
    train_dit.remote()

if __name__ == "__main__":
    main()