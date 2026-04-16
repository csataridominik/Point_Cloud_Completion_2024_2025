import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming furthest_point_sample and gather_points are correctly imported from PointAttN
from PointAttN import furthest_point_sample, gather_points 
# If you don't have PointAttN.py, you'll need to define/import these:
# def furthest_point_sample(xyz, N_out): ...
# def gather_points(x, idx): ...


# === Positional / Time Embedding ===
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

# === Cross-Attention Module ===
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout, batch_first=False) # batch_first=False is the PyTorch default for MultiheadAttention when input is (Seq, Batch, Feature)
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

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        # src1 (Query): [B, C_in, N_q] -> [B, C_out, N_q]
        # src2 (Key/Value): [B, C_in, N_k] -> [B, C_out, N_k]
        
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        # Reshape to (Seq, Batch, Feature) for MultiheadAttention
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1) # [N_q, B, C_out]
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1) # [N_k, B, C_out]

        # Pre-attention normalization (optional, but in original code)
        src1_norm = self.norm13(src1)
        src2_norm = self.norm13(src2)

        # Cross Attention
        src12 = self.multihead_attn1(query=src1_norm, # Use normalized query
                                     key=src2_norm,   # Use normalized key
                                     value=src2)[0]   # Use un-normalized value (common alternative)

        # Skip connection + Post-attention normalization
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        # Feed Forward Network
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        # Reshape back to [B, C_out, N_q]
        src1 = src1.permute(1, 2, 0)

        return src1

# === Helper Blocks ===

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


class PCTEncoderBlock(nn.Module):
    """Wraps cross_transformer for the encoder with time embedding addition."""
    def __init__(self, in_dim, out_dim, time_mlp_proj):
        super().__init__()
        # Initialize the attention part
        self.attn = cross_transformer(in_dim, out_dim, dim_feedforward=out_dim * 4)
        # Accept the pre-initialized time projection module
        self.time_mlp_proj = time_mlp_proj 
    
    def forward(self, q, k, time_emb):
        # q: [B, C_q, N_q], k: [B, C_k, N_k], time_emb: [B, T_dim]
        h = self.attn(q, k)
        # Add time embedding
        h = h + self.time_mlp_proj(time_emb).unsqueeze(-1)
        return h

# === REVISED Decoder Block with Feature Replication ===
class ReplicationDecoderBlock(nn.Module):
    """
    Implements upsampling via feature replication (nearest neighbor interpolation) 
    and refinement using skip connections. Avoids costly geometric interpolation.
    """
    def __init__(self, in_dim_coarse, in_dim_skip, out_dim, time_proj, up_factor=4):
        super().__init__()
        self.up_factor = up_factor
        
        # Accept the pre-initialized time projection module
        self.time_proj = time_proj 
        
        # 1. MLP to combine replicated features and skip features
        # Input: C_coarse (replicated) + C_skip
        self.mlp1 = MLPBlock(in_dim_coarse + in_dim_skip, out_dim)
        self.mlp2 = MLPBlock(out_dim, out_dim)

    def forward(self, x_coarse, skip, time_emb):
        # x_coarse: [B, C_coarse, N_coarse]
        # skip: [B, C_skip, N_skip] where N_skip is the target resolution

        # 1. Replication (Nearest Neighbor interpolation is a memory-safe way to replicate)
        N_target = skip.shape[-1]
        x_replicated = F.interpolate(x_coarse, size=N_target, mode='nearest')
        
        # 2. Concatenate replicated features with skip connection
        x = torch.cat([x_replicated, skip], dim=1)
        
        # 3. Refine features
        h = self.mlp1(x)
        # Add time embedding
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.mlp2(h)
        return h # [B, out_dim, N_skip]

# === Main UNet Model ===
class PointDiffusionUNet(nn.Module):
    def __init__(self, hidden_dim=64, time_emb_dim=128, in_channels=4, N_in_partial=4*1024, N_out_missing=1024*4):
        super().__init__()
        
        self.N_in = N_in_partial + N_out_missing # 20480
        self.N_in_partial = N_in_partial
        self.N_out_missing = N_out_missing
        D = hidden_dim
        
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Time Embedding Projection MLPs (7 total)
        # Indices: 0, 1, 2 (Enc), 3 (Mid), 4, 5 (Dec), 6 (Final)
        dims = [D * 2, D * 4, D * 8, D * 8, D * 4, D * 2, D]
        self.time_mlps = nn.ModuleList([
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            for dim in dims
        ])

        # 1. Feature Lift
        self.feat_lift = nn.Sequential(
            nn.Conv1d(in_channels, D, 1),
            nn.GroupNorm(4, D),
            nn.GELU(),
            nn.Conv1d(D, D, 1) # [B, D, 20480] (h0)
        )

        # 2. PCT-based Encoder (Uses time_mlps[0], [1], [2])
        self.enc1 = PCTEncoderBlock(D, D * 2, self.time_mlps[0])  # 20480 -> 5120 (h1)
        self.enc2 = PCTEncoderBlock(D * 2, D * 4, self.time_mlps[1]) # 5120 -> 1280 (h2)
        self.enc3 = PCTEncoderBlock(D * 4, D * 8, self.time_mlps[2]) # 1280 -> 320 (h3)

        # 3. Bottleneck (Uses time_mlps[3] in forward pass)
        self.mid = cross_transformer(D * 8, D * 8)

        # 4. Decoder (Uses time_mlps[4], [5])
        # Dec 3: N_coarse=320, N_skip=1280. Time proj: self.time_mlps[4] (D*4 channels)
        self.dec3 = ReplicationDecoderBlock(D * 8, D * 4, D * 4, self.time_mlps[4], up_factor=4) 
        # Dec 2: N_coarse=1280, N_skip=5120. Time proj: self.time_mlps[5] (D*2 channels)
        self.dec2 = ReplicationDecoderBlock(D * 4, D * 2, D * 2, self.time_mlps[5], up_factor=4) 
        
        # 5. Final Stage (Uses time_mlps[6] in forward pass)
        # Refinement for the skip connection to the target resolution (16384)
        self.h0_target_skip = nn.Sequential(
             nn.Conv1d(D, D, 1),
             nn.GroupNorm(4, D),
             nn.SiLU(),
        )

        # Final Replication Block (5120 -> 16384 refinement)
        # Input channel count: 2D (replicated d2) + D (refined h0_target skip)
        self.final_replication_head = nn.Sequential(
            nn.Conv1d(D * 2 + D, D, 1), 
            nn.GroupNorm(4, D),
            nn.GELU(),
            nn.Conv1d(D, D, 1) # Final refinement layer
        )

        # 6. Output Head (D channels -> 3 xyz channels)
        self.out_head = nn.Sequential(
            nn.Conv1d(D, D // 2, 1),
            nn.GELU(),
            nn.Conv1d(D // 2, 3, 1) # Predict per-point noise [B, 3, 16384]
        )


    def forward(self, x, t):
        B, _, N = x.shape
        xyz = x[:, :3, :] # [B, 3, 20480]

        t_emb = self.time_emb(t)
        
        # --- Encoder Path (Downsampling) ---
        h0 = self.feat_lift(x) # [B, D, 20480]
        
        # Split features for final skip connection (only for the 16384 missing points)
        h0_target = h0[:, :, self.N_in_partial:] # [B, D, 16384]

        N1, N2, N3 = N // 4, N // 16, N // 64
        
        # Enc 1: 20480 -> 5120
        idx1 = furthest_point_sample(xyz.transpose(1, 2).contiguous(), N1)
        h1_q = gather_points(h0, idx1)  
        h1 = self.enc1(h1_q, h0, t_emb) # [B, 2D, 5120]
        
        # Enc 2: 5120 -> 1280
        xyz1_sampled = gather_points(xyz, idx1) 
        idx2 = furthest_point_sample(xyz1_sampled.transpose(1, 2).contiguous(), N2)
        h2_q = gather_points(h1, idx2) 
        h2 = self.enc2(h2_q, h1, t_emb) # [B, 4D, 1280]

        # Enc 3: 1280 -> 320
        xyz2_sampled = gather_points(xyz1_sampled, idx2)
        idx3 = furthest_point_sample(xyz2_sampled.transpose(1, 2).contiguous(), N3)
        h3_q = gather_points(h2, idx3)
        h3 = self.enc3(h3_q, h2, t_emb) # [B, 8D, 320]

        # Bottleneck (320 points)
        mid = self.mid(h3, h3) + self.time_mlps[3](t_emb).unsqueeze(-1) # [B, 8D, 320]

        # --- Decoder Path (Feature Replication) ---
        
        # Dec 3: 320 -> 1280 
        d3 = self.dec3(mid, h2, t_emb) # [B, 4D, 1280]

        # Dec 2: 1280 -> 5120 
        d2 = self.dec2(d3, h1, t_emb) # [B, 2D, 5120]
        
        # Final Targeted Replication (5120 -> 16384)
        d2_replicated = F.interpolate(d2, size=self.N_out_missing, mode='nearest') # [B, 2D, 16384]
        
        # Refine target skip connection
        h0_skip = self.h0_target_skip(h0_target) # [B, D, 16384]

        # Concatenate and refine
        d1_in = torch.cat([d2_replicated, h0_skip], dim=1) # [B, 3D, 16384]
        
        d1 = self.final_replication_head(d1_in) # [B, D, 16384]
        
        # Add the final time embedding projection (Index 6)
        d1 = d1 + self.time_mlps[6](t_emb).unsqueeze(-1) 

        # --- Output ---
        out = self.out_head(d1)  # predicted noise [B, 3, 16384]
        return out


# === Diffusion Model Wrapper (Optional but useful for training) ===

class DiffusionModel(nn.Module):
    def __init__(self, time_steps=1000, N_in=1024):
        super().__init__()

        self.time_steps = time_steps
        self.N_in = N_in

        # Ensure cosine_beta_schedule is defined or imported
        # self.betas = cosine_beta_schedule(1e-4, 0.02, time_steps)
        # self.alphas = 1 - self.betas
        # self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        
        # Dummy schedule initialization for code completeness
        self.register_buffer('alpha_hats', torch.ones(time_steps))

        # U-Net with upsampling
        self.model = PointDiffusionUNet()
    
    def add_noise(self, x, ts):
        noise = torch.randn_like(x)
        out = []
        for i, t in enumerate(ts):
            a_hat = self.alpha_hats[t]
            out.append(
                torch.sqrt(a_hat) * x[i] + torch.sqrt(1 - a_hat) * noise[i]
            )
        return torch.stack(out), noise

    def forward(self, x_noised, t):
        return self.model(x_noised, t)

# Diffusion Helper Function
def cosine_beta_schedule(beta_start, beta_end, num_timesteps, s=0.002):
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)

    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

    alphas_cumprod = f(x) / f(torch.tensor([0.0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, beta_start, beta_end)

if __name__ == "__main__":
    # Ensure furthest_point_sample and gather_points are available for testing
    def furthest_point_sample(xyz, N_out):
        # Dummy implementation for testing
        return torch.randint(0, xyz.shape[1], (xyz.shape[0], N_out), device=xyz.device)

    def gather_points(x, idx):
        # Dummy implementation for testing
        idx = idx.unsqueeze(1).repeat(1, x.shape[1], 1)
        return torch.gather(x, 2, idx)
    
    # Test the new model structure
    B = 2
    N_partial = 4096
    N_missing = 16384
    N_total = N_partial + N_missing # 20480 points
    D_feat = 64
    T_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PointDiffusionUNet(hidden_dim=D_feat, N_in_partial=N_partial, N_out_missing=N_missing).to(device)

    # Input: [B, 4, 20480] (xyz, mask)
    x = torch.randn(B, 4, N_total, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    
    print("Running forward pass...")
    with torch.no_grad():
        noise_pred = model(x, t)

    print("--- Test Results ---")
    print(f"Output shape: {noise_pred.shape}")
    print(f"Expected shape: [{B}, 3, {N_missing}]")
    print("Test successful!" if noise_pred.shape == torch.Size([B, 3, N_missing]) else "Test FAILED!")
    print(f"Parameter count (M): {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")