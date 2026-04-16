import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Utils ---------
from pytorch3d.loss import chamfer_distance
def calc_cd(pc1,pc2,norm_=2):
    assert pc1.dim() == 3 and pc2.dim() == 3 and pc1.size(2) == 3 and pc2.size(2) == 3, f"Shape mismatch at calc_cd() -> {pc1.shape} and {pc2.shape}"
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)
    return loss

def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)
    return: indices of sampled points (B, npoint)
    """
    # Use naive FPS for simplicity; replace with a fast GPU version for large clouds
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class pcn_encoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_coarse = 1024
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

    def forward(self,xyz,feat_g):
        B,_,N = xyz.shape

        feature = self.first_conv(xyz)#.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feat_g_expanded = feat_g.expand(-1, -1, N)  # (B, 512, N)

        feature = torch.cat(
            [feature_global.expand(-1, -1, N), feature,feat_g_expanded],
            dim=1
        )

        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]   
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                         

        return coarse

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
        
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1) #N//8
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine

from pytorch3d.ops import sample_farthest_points
from pytorch3d.loss import chamfer_distance
def calc_cd(pc1,pc2,norm_=2):
    assert pc1.dim() == 3 and pc2.dim() == 3 and pc1.size(2) == 3 and pc2.size(2) == 3, f"Shape mismatch at calc_cd() -> {pc1.shape} and {pc2.shape}"
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)
    return loss
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    assert xyz.dim() == 3 and xyz.size(2) == 3, f"fps input must be [B,N,3], got {xyz.shape} in def furthest_point_sample. "
    _,fps_points_idx = sample_farthest_points(xyz, K=npoint)
    return fps_points_idx

def gather_points(features, idx):

    """
    Gathers features at the given indices.
    Args:
        features: [B, C, N]
        idx: [B, npoint]
    Returns:
        new_features: [B, C, npoint]
    """
    B, C, N = features.shape
    assert features.dim() == 3, 'Dimension mismatch in gather_points()'
    B, C, N = features.shape
    assert idx.dim() == 2 and idx.shape[0] == B, f'Shape mismatch in gather_points() with shape: {[B,C,N]}'
    
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    new_features = torch.gather(features, 2, idx_expanded)
    return new_features

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    
    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


        self.encoder = PCT_encoder()

        self.anchors = 16
        self.neighborhood = 512

        self.pcn_generate = pcn_encoder(latent_dim=1024)

    def regions(self, new_x, anchors):
        """
        Args:
            new_x:   [B, 3, N] full point cloud
            anchors: [B, 3, A] anchor points

        Returns:
            regions: [B, A, 3, K] KNN neighborhoods per anchor
        """

        B, _, N = new_x.shape
        _, _, A = anchors.shape
        K = self.neighborhood

        # Convert to [B, N, 3] and [B, A, 3]
        xyz = new_x.transpose(1, 2).contiguous()
        anchor_xyz = anchors.transpose(1, 2).contiguous()

        # KNN: get indices [B, A, K]
        knn_idx = knn_point(K, xyz, anchor_xyz)

        # Gather points
        # Expand xyz for gather
        xyz_expand = xyz.unsqueeze(1).expand(-1, A, -1, -1)  # [B, A, N, 3]
        idx_expand = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

        grouped_xyz = torch.gather(xyz_expand, 2, idx_expand)  # [B, A, K, 3]

        # Reorder to [B, A, 3, K]
        regions = grouped_xyz.permute(0, 1, 3, 2).contiguous()

        return regions
  

    def loss(self,coarse,fine,gt):
        
            loss1 = calc_cd(fine.contiguous(), gt.contiguous(),norm_=1)
        
            gt_coarse = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, coarse.shape[1])).transpose(1, 2).contiguous()
            loss2 = calc_cd(coarse.contiguous(), gt_coarse.contiguous(),norm_=1)

            total_train_loss = loss1.mean() + loss2.mean()
            #return fine1, loss2, total_train_loss
            return total_train_loss

    # Data comes as [-1,1]
    def forward(self, x, gt=None, is_training=True):
        
        feat_g, coarse = self.encoder(x)
        #print(f'This is encoder out feat_g: {feat_g.shape}') # [16, 512, 1]
        
        #print(f'This is encoder out coarse: {coarse.shape}') # ([16, 3, 256]), this is actually new_x
        new_x = torch.cat([x,coarse],dim=2)
        anchors = gather_points(new_x, furthest_point_sample(coarse.transpose(1, 2).contiguous(), self.anchors))
        
        closest_regions = self.regions(new_x,anchors)#.transpose(3, 2).contiguous() # [b, 16, 512, 3]

        #new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))
        
        fine = torch.empty(
            gt.shape[0], gt.shape[1], 3, # [b,16384,3]
            device=coarse.device,
            dtype=coarse.dtype
        )

        for anchor in range(self.anchors):
            patch = self.pcn_generate(
                closest_regions[:, anchor, :, :], feat_g
            )  

            fine[:, anchor*patch.shape[1]:(anchor+1)*patch.shape[1], :] = patch
            
        coarse = coarse.transpose(1, 2).contiguous()
        
        loss = self.loss(coarse,fine,gt)
        if is_training:
            return loss
        else:
            return fine, loss
'''
# Yet to try out parts....--------------------------------------------------------------------

class PatchRefiner(nn.Module):
    def __init__(self, latent_dim=1024, num_points_per_patch=1024):
        super().__init__()
        # 1. Mini-PointNet to encode the patch geometry
        self.conv1 = nn.Sequential(nn.Conv1d(3, 128, 1), nn.ReLU(), nn.Conv1d(128, 256, 1))
        
        # 2. Attention to mix information between patches (The Coherence Maker)
        # d_model=512 + 256 (feat_g + local_feat)
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
        # 3. Final Decoder
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * num_points_per_patch)
        )

    def forward(self, regions, feat_g, anchor_centers):
        """
        regions: [B, A, 3, K] (Global coordinates of neighbors)
        feat_g: [B, 512, 1] (Global feature)
        anchor_centers: [B, A, 3] (Where the patches are)
        """
        B, A, _, K = regions.shape
        
        # --- Step 1: Normalize Coordinates (Huge boost for patch quality) ---
        # Subtract anchor center from region points. 
        # The network learns "shape" easier in local coords.
        centers_expanded = anchor_centers.unsqueeze(-1) # [B, A, 3, 1]
        local_regions = regions - centers_expanded      # [B, A, 3, K]
        
        # Collapse B and A for batch processing
        local_regions = local_regions.view(B * A, 3, K)
        
        # Extract local features
        feat = self.conv1(local_regions)                # [B*A, 256, K]
        feat = torch.max(feat, 2)[0]                    # [B*A, 256]
        feat = feat.view(B, A, 256)                     # [B, A, 256]
        
        # --- Step 2: Inject Global Context ---
        feat_g_expanded = feat_g.squeeze(-1).unsqueeze(1).expand(-1, A, -1) # [B, A, 512]
        
        # Concatenate: [Local Shape info] + [Global Object info]
        combined_feat = torch.cat([feat, feat_g_expanded], dim=2) # [B, A, 768]
        
        # --- Step 3: Coherence Mixing (The "For Loop" replacement) ---
        # The anchors talk to each other here.
        # "I am the left headlight, you are the grille, let's align."
        mixed_feat, _ = self.attn(combined_feat, combined_feat, combined_feat)
        
        # Residual connection is often good here
        mixed_feat = mixed_feat + combined_feat
        
        # --- Step 4: Decode ---
        # We can now process all anchors in parallel (no for loop needed!)
        points = self.mlp(mixed_feat) # [B, A, 3*1024]
        points = points.view(B, A, 1024, 3)
        
        # Add the centers back to move points to global space
        final_points = points + anchor_centers.unsqueeze(2)
        
        return final_points
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = PCT_encoder()

        self.anchors = 16
        self.neighborhood = 512

        self.pcn_generate = PatchRefiner(latent_dim=1024)

    def regions(self, new_x, anchors):
        """
        Args:
            new_x:   [B, 3, N] full point cloud
            anchors: [B, 3, A] anchor points

        Returns:
            regions: [B, A, 3, K] KNN neighborhoods per anchor
        """

        B, _, N = new_x.shape
        _, _, A = anchors.shape
        K = self.neighborhood

        # Convert to [B, N, 3] and [B, A, 3]
        xyz = new_x.transpose(1, 2).contiguous()
        anchor_xyz = anchors.transpose(1, 2).contiguous()

        # KNN: get indices [B, A, K]
        knn_idx = knn_point(K, xyz, anchor_xyz)

        # Gather points
        # Expand xyz for gather
        xyz_expand = xyz.unsqueeze(1).expand(-1, A, -1, -1)  # [B, A, N, 3]
        idx_expand = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

        grouped_xyz = torch.gather(xyz_expand, 2, idx_expand)  # [B, A, K, 3]

        # Reorder to [B, A, 3, K]
        regions = grouped_xyz.permute(0, 1, 3, 2).contiguous()

        return regions
  

    def loss(self,coarse,fine,gt):
        
            loss1 = calc_cd(fine.contiguous(), gt.contiguous(),norm_=1)
        
            gt_coarse = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, coarse.shape[1])).transpose(1, 2).contiguous()
            loss2 = calc_cd(coarse.contiguous(), gt_coarse.contiguous(),norm_=1)

            total_train_loss = loss1.mean() + loss2.mean()
            #return fine1, loss2, total_train_loss
            return total_train_loss,loss1

    # Data comes as [-1,1]
    def forward(self, x, gt=None, is_training=True):
        
        feat_g, coarse = self.encoder(x)
        new_x = torch.cat([x,coarse],dim=2)
        anchors = gather_points(new_x, furthest_point_sample(coarse.transpose(1, 2).contiguous(), self.anchors))
        
        closest_regions = self.regions(new_x,anchors)#.transpose(3, 2).contiguous() # [b, 16, 512, 3]

        anchor_centers_input = anchors.transpose(1, 2).contiguous()
        
        # Pass everything at once. No For Loop!
        # The Refiner handles the coherence internally via Attention.
        fine_patches = self.pcn_generate(closest_regions, feat_g, anchor_centers_input)
        
        # fine_patches is [B, 16, 1024, 3]
        # Flatten to [B, 16384, 3]
        fine = fine_patches.view(x.shape[0], -1, 3)

        coarse = coarse.transpose(1, 2).contiguous()
        
        
        if is_training:
            loss = self.loss(coarse,fine,gt)
            return loss
        else:
            _,loss = self.loss(coarse,fine,gt)
            return fine, loss


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    BATCH_SIZE = 2
    INPUT_POINTS = 4096
    OUTPUT_POINTS = 16384
    PATCH_POINTS = 256
    T = OUTPUT_POINTS // PATCH_POINTS
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {DEVICE}")
    print(f"Iterations T: {T}")

    # -----------------------------
    # Create dummy input
    # -----------------------------
    dummy_input = torch.randn(BATCH_SIZE, 3,INPUT_POINTS).to(DEVICE)
    
    dummy_gt = torch.randn(BATCH_SIZE, 16384,3).to(DEVICE)
    print("Input shape:", dummy_input.shape)

    # -----------------------------
    # Initialize model
    # -----------------------------
    model = Model().to(DEVICE)

    # -----------------------------
    # Count parameters
    # -----------------------------
    total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {total_params:,}")

    # -----------------------------
    # Forward pass
    # -----------------------------
    model.eval()
    with torch.no_grad():
        output,cd_loss = model(dummy_input,dummy_gt,is_training=False)

    # -----------------------------
    # Output checks
    # -----------------------------
    print("Generated output shape:", output.shape)

    expected_shape = (BATCH_SIZE, OUTPUT_POINTS, 3)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"

    print("✅ Forward pass successful!")
    print("✅ Output size matches expected 16384 points")

if __name__ == "__main__":
    main()
