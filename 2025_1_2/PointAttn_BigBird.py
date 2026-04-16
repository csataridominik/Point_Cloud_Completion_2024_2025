from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from pytorch3d.ops import sample_farthest_points
from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from xformers.components.attention.sparsity_config import BigBirdSparsityConfig

from xformers.components.attention import ScaledDotProduct
import torch.nn as nn
import xformers.ops as xops

from hilbert_z_order import encode_hilbert,encode_z


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

def calc_cd(pc1,pc2,norm_=2):
    assert pc1.dim() == 3 and pc2.dim() == 3 and pc1.size(2) == 3 and pc2.size(2) == 3, f"Shape mismatch at calc_cd() -> {pc1.shape} and {pc2.shape}"
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)
    return loss


# configure sparsity

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

class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=1, dim_feedforward=1024, dropout=0.0):
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

class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1,refine_step=0):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        """ 
        self.sa1 = self_transformer(channel*2, 512, nhead=8, block_size=32,
                            num_sliding_window_blocks=20, num_random_blocks=18, num_global_blocks=4)

        self.sa2 = self_transformer(512,512)
        self.sa3 = self_transformer(512,channel*ratio)
        """
        if refine_step == 1:
            self.sa1 = self_transformer(channel*2, 512,
                                block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)

            self.sa2 = self_transformer(512, 512, 
                                        block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)

            self.sa3 = self_transformer(512, channel*ratio,
                                        block_size=64, num_sliding_window_blocks=4,
                                num_random_blocks=2, num_global_blocks=1)
        elif refine_step == 2:
            self.sa1 = self_transformer(channel*2, 512,
                                block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)

            self.sa2 = self_transformer(512, 512, 
                                        block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)

            self.sa3 = self_transformer(512, channel*ratio,
                                        block_size=64, num_sliding_window_blocks=16,
                                num_random_blocks=8, num_global_blocks=1)
        
        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

    def forward(self, x, coarse,feat_g):
        
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        # y = [2,128,512]
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        # feat_g = [2,128,1]
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)
        # y0 = [2, 256,512]
        print(f'This is y0 here decode: {y0.shape}')
        y1 = self.sa1(y0, y0)
        print(f'This is y0 here decode after pcn: {y1.shape}')
        # y1 = [2, 512,512]
        
        y2 = self.sa2(y1, y1)
        # y2 = [2, 512,512]     
        y3 = self.sa3(y2, y2)
        print(f'This is y3: {y3.shape}')
        
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)
        # y3 = [2, 128,2048]
        y_up = y.repeat(1,1,self.ratio)
        # y_up = [2, 128,2048]
        y_cat = torch.cat([y3,y_up],dim=1)
        # y_cat = [2, 256,2048]
        y4 = self.conv_delta(y_cat)
        # y4 = [2, 128,2048]
        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)
        # x = [2, 3,2048]

        return x, y3
    
'''
class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = self_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = self_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = self_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()

        self.sa0_d = self_transformer(channel*8,channel*8)
        self.sa1_d = self_transformer(channel*8,channel*8)
        self.sa2_d = self_transformer(channel*8,channel*8)

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
        
        x2_d = (self.sa2_d(x1_d, x1_d))
        
        x2_d = x2_d.reshape(batch_size,self.channel*4,-1) #N//16 works = 128, actual size [B,256,256]!!
        
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        
        return x_g, fine
'''
from PCT_Encoding import PCT

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

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        else:
            ValueError('dataset is not exist')

        self.encoder = PCT_encoder()
        #self.encoder = PCT(samples=[1024, 512])

        self.refine = PCT_refine(ratio=4,refine_step=1)
        self.refine1 = PCT_refine(ratio=8,refine_step=2)

        self.anchor_points = 64

        #self.create_coarse = create_feat_g()

    def encode(self,x,space,curve='z'):
        x = x.permute(0, 2, 1)
        # should be [b,points,3]
        if curve == 'hilbert':
            x_encoded = encode_hilbert(x, space_size=space, convention="xyz")
        elif curve == 'z':
            x_encoded = encode_z(x, space_size=space, convention="xyz")

        sorted_encoded, sort_idx = torch.sort(x_encoded, dim=1)
        sorted_xyz = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
        sorted_xyz = sorted_xyz.permute(0, 2, 1)
        return sorted_xyz
    # Data comes as [-1,1]
    def forward(self, x, gt=None, is_training=True):
        
        x = self.encode(x,x.shape[2]) #[b,3,points]
        
        # [ORDER X HERE]
        #feat_g, coarse = self.encoder(x)
        feat_g, coarse = self.encoder(x)
        print(f'feat_g: {feat_g.shape}')

        print(f'coarse: {coarse.shape}')
        #coarse = self.create_coarse(feat_g)

        #print(f'This is encoder out feat_g: {feat_g.shape}') # [16, 512, 1]
               
        new_x = torch.cat([coarse,x],dim=2)

        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512-self.anchor_points))
        
        coarse_anchors = gather_points(coarse, furthest_point_sample(coarse.transpose(1, 2).contiguous(), self.anchor_points))

        # [ORDER NEW_X HERE]
        #new_x = self.encode(new_x,new_x.shape[2])
        
        new_x = torch.cat([coarse_anchors,new_x],dim=2)
        
        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:

            loss3 = calc_cd(fine1.contiguous(), gt.contiguous(),norm_=1)
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()
            
            loss2 = calc_cd(fine.contiguous(), gt_fine1.contiguous(),norm_=1)
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

            loss1 = calc_cd(coarse.contiguous(), gt_coarse.contiguous(),norm_=1)

            #total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            #return fine1, loss2, total_train_loss
            return total_train_loss
        else:

            cd_p = calc_cd(fine1, gt,norm_=2)
            cd_t = calc_cd(fine1, gt,norm_=1)
            cd_p_coarse = calc_cd(coarse, gt,norm_=2)
            cd_t_coarse = calc_cd(coarse, gt,norm_=1)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}

class create_feat_g(nn.Module):
    def __init__(self, channel=64):
        super(create_feat_g, self).__init__()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4*2, 64, kernel_size=1)
        
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)
        
        self.relu = nn.GELU()

    def forward(self, x_g):
        print(f'This is shape of x_g: {x_g.shape}')
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))

        fine = self.conv_out(self.relu(self.conv_out1(x)))

        return fine

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        #self.sa1_1 = self_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        #self.sa2_1 = self_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        #self.sa3_1 = self_transformer((channel)*8,channel*8)

        # 60% coverage for full attention:
        self.sa1_1 = self_transformer(channel*2, channel*2, 
                              block_size=64, num_sliding_window_blocks=7,
                              num_random_blocks=3, num_global_blocks=1) # 1024 tokens,-> 

        self.sa2_1 = self_transformer(channel*4, channel*4, 
                                    block_size=32, num_sliding_window_blocks=7,
                                    num_random_blocks=3, num_global_blocks=1)

        self.sa3_1 = self_transformer(channel*8, channel*8, 
                                    block_size=16, num_sliding_window_blocks=7,
                                    num_random_blocks=3, num_global_blocks=1)


        # new:
        #self.sa4 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()
        
        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)
        """ # inside your encoder __init__:
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
         """
        
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

        self.global_points = GlobalPoints(in_channels=512, num_points=512)

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
        points = gather_points(points, idx_2) # This is also changed!!!!!!
        # points = gather_points(points, idx_2)

        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous() #[b,512,N//16]

        # seed generator
        # maxpooling
        #x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1) #[b,512,1]
        x3 = x3.transpose(1,2)

        x_g = self.global_points(x3)
        x_g = x_g.transpose(1,2)
        
        
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,-1) #N//8

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine

if __name__ == "__main__":

    # ---- Dummy args ----
    class Args:
        def __init__(self):
            self.dataset = 'pcn'  # or 'c3d'
    args = Args()

    # ---- Create model ----
    model = Model(args)
    model.cuda()  # remove this line if no GPU

    # ---- Dummy input ----
    B = 2        # batch size
    N = 2048*2     # number of input points
    M = 4096*2     # number of ground truth points

    x = torch.rand(B, 3, N).cuda()   # input partial point cloud
    gt = torch.rand(B, M, 3).cuda()  # ground truth complete point cloud

    # ---- Training mode ----
    model.train()
    total_loss = model(x, gt, is_training=True)
    print("Training mode:")
    print(f"total loss is: {total_loss}")
    # ---- Evaluation mode ----
    model.eval()
    with torch.no_grad():
        out = model(x, gt, is_training=False)
    print("\nEvaluation mode:")
    print(f"Coarse output shape: {out['out1'].shape}")
    print(f"Fine output shape: {out['out2'].shape}")
    print(f"Chamfer coarse (pred/true): {out['cd_p_coarse'].item():.6f}, {out['cd_t_coarse'].item():.6f}")
    print(f"Chamfer fine (pred/true): {out['cd_p'].item():.6f}, {out['cd_t'].item():.6f}")
