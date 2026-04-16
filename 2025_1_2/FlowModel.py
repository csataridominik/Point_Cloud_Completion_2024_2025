import torch
import torch.nn as nn
import torch.nn.functional as F

class FromBlogPost(nn.Module):  # Neural network to learn the time-dependent velocity field f(x, t)
  def __init__(self, input_dim=2, time_embed_dim=64):
    super().__init__()
    
    # Small MLP to embed the time scalar t into a higher-dimensional space
    self.time_embed = nn.Sequential(
        nn.Linear(1, time_embed_dim),
        nn.SiLU(),                     # Activation function: Sigmoid Linear Unit
        nn.Linear(time_embed_dim, time_embed_dim)
    )

    # Main network to predict velocity, given (x, embedded t)
    self.net = nn.Sequential(
        nn.Linear(input_dim + time_embed_dim, 128),  # Input: concatenated x and t embedding
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, input_dim)  # Output: predicted velocity (same dimension as x)
    )

  def forward(self, x, t):
    # Embed time t (shape: [batch_size, 1]) into a higher-dimensional vector
    t_embed = self.time_embed(t)

    # Concatenate position x and time embedding along the last dimension
    xt = torch.cat([x, t_embed], dim=-1)

    # Pass through the network to predict the velocity at (x, t)
    return self.net(xt)


# --- Time Embedding (same as before, but higher dim for images) ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.net(t)

# --- A Simple Residual Block for the U-Net ---
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

# --- A Compact U-Net for Flow Matching ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_embed_dim=128):
        super().__init__()
        self.down1 = ResBlock(in_channels, base_channels, time_embed_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_embed_dim)
        self.mid = ResBlock(base_channels * 2, base_channels * 2, time_embed_dim)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_embed_dim)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t_emb):
        h1 = self.down1(x, t_emb)
        h2 = F.avg_pool2d(h1, 2)
        h2 = self.down2(h2, t_emb)
        h2 = self.mid(h2, t_emb)
        h = F.interpolate(h2, scale_factor=2, mode="nearest")
        h = self.up1(h, t_emb)
        return self.out_conv(h)

# --- Final FlowModel wrapper ---
class FlowModel(nn.Module):
    def __init__(self, in_channels=3, time_embed_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.unet = UNet(in_channels, base_channels=64, time_embed_dim=time_embed_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.unet(x, t_emb)




def flow_matching_loss(model, x0, x1, t):
  # Compute the interpolated point along the trajectory for each t
  t_expanded = t[:, :, None, None]
  xt = (1 - t_expanded) * x0 + t_expanded * x1

  # Compute the ground truth velocity vector (constant across trajectory)
  v_target = x1 - x0

  # Predict the velocity at point (x(t), t) using the model
  v_pred = model(xt, t)

  # Compute squared error between predicted and true velocity at each sample
  # Then average over the entire batch
  return ((v_pred - v_target) ** 2).mean()