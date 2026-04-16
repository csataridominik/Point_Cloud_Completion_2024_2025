import torch
import torch.nn as nn
import torch.nn.functional as F
from PointAttN import furthest_point_sample, gather_points

# For the encoder:
class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()
        
        # seed generator
        # maxpooling

        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        print(f'x_g.shape: {x_g.shape}')
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1) #N//8

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine

# Positional / Time Embedding
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] integer timesteps
        returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device) *
            (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

# Basic MLP block
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.GroupNorm(4, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)

# U-Net Encoder Block
class PointEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.mlp1 = MLPBlock(in_dim, out_dim)
        self.mlp2 = MLPBlock(out_dim, out_dim)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )

    def forward(self, x, time_emb):
        h = self.mlp1(x)
        h = h + self.time_mlp(time_emb).unsqueeze(-1)
        h = self.mlp2(h)
        return h


# U-Net Decoder Block
class PointDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.mlp1 = MLPBlock(in_dim, out_dim)
        self.mlp2 = MLPBlock(out_dim, out_dim)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim)
        )

    def forward(self, x, skip, time_emb):
        x = torch.cat([x, skip], dim=1)
        h = self.mlp1(x)
        h = h + self.time_mlp(time_emb).unsqueeze(-1)
        h = self.mlp2(h)
        return h


# Upsampling Module: expands features 4×
class PointUpsample(nn.Module):
    """
    Turns feature map [B, C, N] into [B, C, N * up_factor]
    """
    def __init__(self, in_dim, up_factor=4):
        super().__init__()
        self.up_factor = up_factor

        # Each original point produces up_factor new points
        self.expand = nn.Conv1d(in_dim, in_dim * up_factor, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, 1),
            nn.SiLU(),
            nn.Conv1d(in_dim // 2, 3, 1),
        )

    def forward(self, x):
        """
        x: [B, C, N]
        -> [B, C, N*up_factor]
        """
        B, C, N = x.shape
        h = self.expand(x)                          # [B, C*up_factor, N]
        h = h.reshape(B, C, self.up_factor * N)     # [B, C, N_out]
        out = self.mlp(h)                           # [B, 3, N_out]
        return out

class PointUNetDiffusion(nn.Module):
    def __init__(self, hidden_dim=128, time_emb_dim=128, in_channels=4):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # first block input channels = in_channels (3 coords + 1 mask)
        self.enc1 = PointEncoderBlock(in_channels, hidden_dim, time_emb_dim)
        self.enc2 = PointEncoderBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.enc3 = PointEncoderBlock(hidden_dim * 2, hidden_dim * 4, time_emb_dim)

        self.mid = PointEncoderBlock(hidden_dim * 4, hidden_dim * 4, time_emb_dim)

        self.dec3 = PointDecoderBlock(hidden_dim * 4 + hidden_dim * 4, hidden_dim * 2, time_emb_dim)
        self.dec2 = PointDecoderBlock(hidden_dim * 2 + hidden_dim * 2, hidden_dim, time_emb_dim)
        self.dec1 = PointDecoderBlock(hidden_dim + hidden_dim, hidden_dim, time_emb_dim)

        # final predict per-point noise (3 channels)
        self.out_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim//2, 3, 1)
        )

    def forward(self, x, t):
        # coords: [B,3,N], mask: [B,1,N], concat -> [B,4,N]
        
        t_emb = self.time_emb(t)

        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(h1, t_emb)
        h3 = self.enc3(h2, t_emb)
        mid = self.mid(h3, t_emb)

        d3 = self.dec3(mid, h3, t_emb)
        d2 = self.dec2(d3, h2, t_emb)
        d1 = self.dec1(d2, h1, t_emb)

        out = self.out_head(d1)  # predicted noise per point [B,3,N]
        return out



# Diffusion Wrapper
def cosine_beta_schedule(beta_start, beta_end, num_timesteps, s=0.002):
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)

    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

    alphas_cumprod = f(x) / f(torch.tensor([0.0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, beta_start, beta_end)


class DiffusionModel(nn.Module):
    def __init__(self, time_steps=1000, N_in=4096):
        super().__init__()

        self.time_steps = time_steps
        self.N_in = N_in

        self.betas = cosine_beta_schedule(1e-4, 0.02, time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        # U-Net with upsampling
        self.model = PointUNetDiffusion()
    """ 
    def add_noise(self, x, ts):
        noise = torch.randn_like(x)
        out = []
        for i, t in enumerate(ts):
            a_hat = self.alpha_hats[t]
            out.append(
                torch.sqrt(a_hat) * x[i] + torch.sqrt(1 - a_hat) * noise[i]
            )
        return torch.stack(out), noise
     """
    def get_alpha_hats(self,t):
        return self.alpha_hats[t]
    def add_noise(self, x0, mask, ts):
        """
        x0: [B,3,N] ground-truth full cloud
        mask: [B,1,N] 1=observed, 0=missing
        ts: [B] timesteps (long)
        returns: x_t, noise (both [B,3,N])
        We only add noise to missing points
        """
        B = x0.shape[0]
        device = x0.device
        noise = torch.randn_like(x0)

        out = []
        for i in range(B):
            t = ts[i].item()
            a_hat = self.alpha_hats[t].to(device)
            sqrt_ahat = torch.sqrt(a_hat)
            sqrt_1_ahat = torch.sqrt(1 - a_hat)

            # observed points remain exact (x_obs)
            x_obs = mask[i] * x0[i]  # [3,N]
            # missing points: forward diffusion
            x_missing_noised = sqrt_ahat * x0[i] + sqrt_1_ahat * noise[i]
            x_t_i = x_obs + (1.0 - mask[i]) * x_missing_noised
            out.append(x_t_i)

        x_t = torch.stack(out, dim=0)
        return x_t, noise


    def forward(self, x_noised, t):
        return self.model(x_noised, t)


if __name__ == "__main__":
    B = 2
    N_in = 4096+ 4096*4
    up_factor = 4
    N_out = N_in * up_factor

    model = DiffusionModel()

    x = torch.randn(B, 4, N_in)
    t = torch.randint(0, 1000, (B,))
    noise_pred = model(x, None, t)

    print("Output shape:", noise_pred.shape)
    print("Expected shape: [B, 3, 16384]")
    print("Parameter count(M):", sum(p.numel() for p in model.parameters())/1_000_000)
