import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points

# ------------------------- Helper functions: --------------------------------
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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

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

# ------------------------- Helper functions end ------------------------------

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout = 0.0):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, out_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, num_patch,dropout=dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dim, dropout=dropout),
        )



    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x    

class GlobalPoints(nn.Module):
    def __init__(self, in_channels=512, num_points=512):
        super().__init__()
        self.num_points = num_points
        
        # Map each feature vector to 3D point
        self.mlp = nn.Sequential(
            #nn.Conv1d(in_channels, 256, 1),
            #nn.BatchNorm1d(256),
            #nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, 1),

             # output 3D coordinates
        )


    def forward(self, x):
        """
        x: [B, C, N] feature map
        returns: [B, 3, N] global points
        """
        return self.mlp(x)


class self_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=1,
                  dim_feedforward=1024, dropout=0.0, block_size=32,
                  num_random_blocks = 6, num_sliding_window_blocks= 6,num_global_blocks=4):
        super().__init__()
        #self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # isntead of this mutlihead i use bigbird:
        self.d_model= d_model
        self.nhead = nhead
        self.dropout=dropout

        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.num_sliding_window_blocks = num_sliding_window_blocks
        self.num_global_blocks = num_global_blocks

        self.sparsity_config = BigBirdSparsityConfig(
            num_heads = nhead,
            block_size = self.block_size,                 # 290 tokens, rougly 14 -> 30 -> 56 coverage...
            num_random_blocks = self.num_random_blocks,
            num_sliding_window_blocks = self.num_sliding_window_blocks,
            num_global_blocks = self.num_global_blocks,
            attention = "bidirectional",
            different_layout_per_head=False
            )
         
        self.attn = ScaledDotProduct(
            dropout=dropout,
            sparsity_config=self.sparsity_config,
        )

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

    def forward(self, src1, src2, if_act=False):


        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.attn(src1,src2,src2)


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1



from xformers.components.attention.sparsity_config import BigBirdSparsityConfig

from xformers.components.attention import ScaledDotProduct
class MLPEncoder(nn.Module):
    def __init__(self,channel=64):
        super().__init__()
         
        #self.sa0_d = cross_transformer(channel*8,channel*8)
        #self.sa1_d = cross_transformer(channel*8,channel*8)
        #self.sa2_d = cross_transformer(channel*8,channel*8)
        
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)
        
        self.activation = nn.GELU()
        self.channel = channel
        
        self.sa0_d = self_transformer(
            d_model=channel*8, d_model_out=channel*8,
            block_size=16,
            num_sliding_window_blocks=6,   # 2 blocks in the local window
            num_random_blocks=1,           # 1 random block
            num_global_blocks=1,           # 1 global block
            dim_feedforward=1024,
            dropout=0.0
        )

        self.sa1_d = self_transformer(
            d_model=channel*8, d_model_out=channel*8,
            block_size=16,
            num_sliding_window_blocks=6,
            num_random_blocks=1,
            num_global_blocks=1,
            dim_feedforward=1024,
            dropout=0.0
        )

        self.sa2_d = self_transformer(
            d_model=channel*8, d_model_out=channel*8, 
        
            block_size=16,
            num_sliding_window_blocks=6,
            num_random_blocks=1,
            num_global_blocks=1,
            dim_feedforward=1024,
            dropout=0.0
        )




    def forward(self,x):
        b = x.shape[0]
        x_g = F.adaptive_max_pool1d(x, 1).view(b, -1).unsqueeze(-1)
        #x_g = self.global_points(x.permute(0,2,1))
        #x_g = x_g.permute(0,2,1)
        x = self.activation(self.ps_adj(x_g))
        x = self.activation(self.ps(x))
        x = self.activation(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(b,self.channel*4,-1) 

        fine = self.conv_out(self.activation(self.conv_out1(x2_d)))
        #fine = self.conv_out(self.activation(self.conv_out1(x.reshape(b,self.channel*4,-1))))
        
        return x_g, fine

class MLPMixer(nn.Module):

    def __init__(self, in_channels=3, dim=256, patch_size=1, pcd_size=4096, depth=1, token_dim=256, channel_dim=512):
        super().__init__()

        self.k = 32

        assert pcd_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  pcd_size#// patch_size)
        
        self.to_patch_embedding = nn.Sequential(
         nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b d n -> b n d')
        ) 
        
        self.to_patch_embedding_asd = nn.Sequential(
            Rearrange('b num_of_patches k c -> b (k c) num_of_patches'),
            nn.Conv1d(self.k * 3, dim, kernel_size=1),
            Rearrange('b d n -> b n d')
        ) 

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim, out_dim = 512,dropout=0.0))

        self.layer_norm = nn.LayerNorm(dim)
        
        self.out = MLPEncoder()

        self.reduce_layer = FeedForward(4096,1024,512,dropout=0.1)

    def forward(self, x):
        """ 
        anchors = gather_points(x, furthest_point_sample(x.transpose(1, 2).contiguous(), self.num_patch))
        anchors = anchors.permute(0, 2, 1)  # → [B, S, 3]

        x_for_knn = x.permute(0, 2, 1)      # → [B, N, 3]

        idx = knn_point(self.k, x_for_knn, anchors)  # now shapes match
        grouped_features = index_points(x_for_knn, idx) """  
        # grouped_features = torch.Size([b, num_of_patches, k, 3])

        #x = self.to_patch_embedding_asd(grouped_features)

        x = self.to_patch_embedding(x)

        #x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        x = x.permute(0, 2, 1)
        x = self.reduce_layer(x)
        x = x.permute(0, 2, 1)
        
        x_mixed = self.layer_norm(x)

        #x = x_mixed.mean(dim=1)

        # Here x_mixed comes out as:  torch.Size([8, 128, 512])
        x_g, course = self.out(x_mixed)
        return x_g,course 

class MixerBlock_decoder(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, ratio=4, dropout=0.):
        super().__init__()
        # Token Mixing MLP with upsampling
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patch * ratio),   # <-- UPSAMPLE TOKENS HERE
            Rearrange('b d (n r) -> b (n r) d', r=ratio),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout=dropout),
        )


    def forward(self, x):
        x = self.token_mix(x)            # [B, N*ratio, C]
        x = x + self.channel_mix(x)
        return x

class MLPMixer_decoder(nn.Module):

    def __init__(self, pcd_size=1024 ,dim=128, depth=1, token_dim=256, channel_dim=512,ratio=4, refine_step=1):
        super().__init__()
        
        self.channel = 128

        self.num_patch= pcd_size
        self.mixer_blocks = nn.ModuleList([])

        for it in range(depth):
            if it == 0:
                self.mixer_blocks.append(MixerBlock(self.channel*ratio, self.num_patch, token_dim, channel_dim,dropout=0.0,out_dim=512))
            else:
                self.mixer_blocks.append(MixerBlock(self.channel*ratio, self.num_patch, token_dim, channel_dim,dropout=0.0,out_dim=512))

        self.layer_norm = nn.LayerNorm(dim)

        # from pct_encoder:

        self.ratio = ratio
        
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_1 = nn.Conv1d(256, self.channel, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.activation = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.conv_delta = nn.Conv1d(self.channel * 2, self.channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(self.channel*ratio, self.channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, self.channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(self.channel, 64, kernel_size=1)

        #self.sa1 = cross_transformer(self.channel*2, self.channel*ratio)
        if refine_step == 1:
            self.sa1 = self_transformer(self.channel*2, 512,
                                block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)

            self.sa2 = self_transformer(512, 512, 
                                        block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)

            self.sa3 = self_transformer(512, self.channel*ratio,
                                        block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)
        elif refine_step == 2:
            self.sa1 = self_transformer(self.channel*2, 512,
                                block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)

            self.sa2 = self_transformer(512, 512, 
                                        block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)

            self.sa3 = self_transformer(512, self.channel*ratio,
                                        block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)
        

    def forward(self, asd,coarse,feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.activation(self.conv_x(coarse)))  # B, C, N
        # y = [2,128,512]

        feat_g = self.conv_1(self.activation(self.conv_11(feat_g)))  # B, C, N
        # feat_g = [2,128,1]
        
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        #y0 = y0.permute(0,2,1)
        # y0 = [2, 256,512]
        
        # ----- Here mlp mixer ------------------
        
        y0 = self.sa1(y0,y0)#.permute(0,2,1)
        y1 = self.sa2(y0, y0)
        y2 = self.sa3(y1, y1).permute(0,2,1)
        for mixer_block in self.mixer_blocks:
            y2 = mixer_block(y2)

        # ----- Here mlp mixer ------------------
        y2 = y2.permute(0,2,1)

        #y0 = torch.ones((batch_size, 256, 256), device="cuda", dtype=torch.float)
        #y0 =  y0.permute(0, 2, 1).contiguous()
        

        y3 = self.conv_ps(y2).reshape(batch_size,-1,N*self.ratio)
        # y3 = [2, 128,2048]
        
        y_up = y.repeat(1,1,self.ratio)
        # y_up = [2, 128,2048]

        y_cat = torch.cat([y3,y_up],dim=1)
        # y_cat = [2, 256,2048]

        y4 = self.conv_delta(y_cat)
        # y4 = [2, 128,2048]

        x = self.conv_out(self.activation(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)
        # x = [2, 3,2048]

        return x,coarse 

   

if __name__ == "__main__":
    img = torch.ones([8, 3, 4096]).to('cuda')

    model = MLPMixer().to('cuda')

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    x_g,course = model(img)

    print(["course :", course.shape])
    
    print(["x_g :", x_g.shape])