import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, ball_query
from pytorch3d.loss import chamfer_distance


# auxiliary functions:

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

def calc_cd(pc1,pc2,norm_=2):
    assert pc1.dim() == 3 and pc2.dim() == 3 and pc1.size(2) == 3 and pc2.size(2) == 3, f"Shape mismatch at calc_cd() -> {pc1.shape} and {pc2.shape}"
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)
    return loss


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


class EncoderBlock(nn.Module):
    def __init__(self,channel=64,input_channel=3):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.act = nn.GELU()

        self.sa1 = cross_transformer(channel,channel)
        self.sa2 = cross_transformer(channel,channel//2)
        self.sa3 = cross_transformer(channel//2,channel//4)

        #for rois(?)-> 256 neighbors...:
        #self.sa4 = cross_transformer(channel//4,channel//8)


    def forward(self,points):
        #print(f'This is points shape: {points.shape}')
        embedded_points = self.act(self.conv1(points))  # B, D, N
        embedded_points = self.conv2(embedded_points)

        x0 = self.sa1(embedded_points,embedded_points)
        x1 = self.sa2(x0,x0)
        x2 = self.sa3(x1,x1)
        
        #x3 = self.sa4(x2,x2) future attention()

        return x2


class EncoderBlock_for_bottleneck(nn.Module):
    def __init__(self,channel=64,input_channel=3):
        super().__init__()

        self.sa1 = cross_transformer(channel,channel//4)

    def forward(self,points):

        x0 = self.sa1(points,points)
        return x0


class BottleneckBlock(nn.Module):
    def __init__(self,channel=64):
        super().__init__()

        self.conv1 = nn.Conv1d(12, 6, kernel_size=1)
        self.conv2 = nn.Conv1d(6, 3, kernel_size=1)
        self.act = nn.GELU()

    def forward(self,points):
        x_g = F.adaptive_max_pool1d(points.permute(0,2,1), 1).view(points.shape[0], -1).unsqueeze(-1)
        fine = self.conv2(self.act(self.conv1(points)))
        return x_g, fine

class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

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
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        return x, y3


from MLP_Mixer_for_dapcn import ChannelMixer

class Model(nn.Module):
    def __init__(self,channel=64,course=512):
        super().__init__()

        self.course = course

        self.ROI1_block = EncoderBlock()
        self.ROI2_block = EncoderBlock()
        self.FPS_course = EncoderBlock()

        self.b = nn.Sequential(
            *[
                #MixerLayer(48,512,2,dropout=0.3)
                ChannelMixer(512,48,2,dropout=0.3)
                for _ in range(4)
            ],
             EncoderBlock_for_bottleneck(channel=48,input_channel=48),
             BottleneckBlock()
        )

        self.refine = PCT_refine(ratio=4)
        self.refine1 = PCT_refine(ratio=8)

    def gather_regions(self, p, a, radius, K=1000, top_k=4, neighborhood=128):
        """
        p: points (B, C, N)
        a: anchors (B, C, num_anchors)
        Returns:
            neighborhoods: (B, top_k * neighborhood, 3)
        """

        B, C, N = p.shape
        device = p.device

        p_xyz = p.permute(0, 2, 1).contiguous()   # (B, N, 3)
        a_xyz = a.permute(0, 2, 1).contiguous()   # (B, A, 3)

        # ---- 1. Ball query to find neighbors for all anchors ----
        _, idx, _ = ball_query(a_xyz, p_xyz, K=K, radius=radius)  # idx: (B, A, K)

        # ---- 2. Count neighbors per anchor ----
        neighbor_counts = (idx != -1).sum(dim=2)  # (B, A)

        # ---- 3. Select top-k anchors per batch ----
        sorted_counts, sorted_idx = torch.sort(neighbor_counts, descending=True)
        topk_idx = sorted_idx[:, :top_k]   # (B, top_k)

        # ---- 4. Gather top-k anchor coordinates ----
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)
        selected_anchors = a_xyz[batch_indices, topk_idx]  # (B, top_k, 3)

        # ---- 5. For each selected anchor, get 'neighborhood' points via KNN ----
        neighbor_idx = knn_point(neighborhood, p_xyz, selected_anchors)  # (B, top_k, neighborhood)

        # ---- 6. Gather the neighbor points ----
        batch_idx = batch_indices.unsqueeze(-1).expand(B, top_k, neighborhood)
        neighborhoods = p_xyz[batch_idx, neighbor_idx]   # (B, top_k, neighborhood, 3)

        # ---- 7. Concatenate neighborhoods ----
        neighborhoods = neighborhoods.reshape(B, top_k * neighborhood, 3)  # (B, top_k*neighborhood, 3)

        return neighborhoods

    def sample_regions(self,points,regions=50, r=0.2):
        
        anchors_idx = furthest_point_sample(points.transpose(1, 2).contiguous(), regions)      
        anchors = gather_points(points, anchors_idx)

        return self.gather_regions(points,anchors,r)

    def forward(self,points,gt,is_training=True):
        b,dim,n = points.shape
        a,b = 0.4,0.4 # NEEDS TO BE FINE-TUNED!!!!!!

        # These are the breanches for parts different segmetns to look into:
        ROI_anchors1_idx = self.sample_regions(points,regions=50,r=a) # [b,3,128*4] 
        ROI_anchors2_idx = self.sample_regions(points,regions=50,r=b) # [b,3,128*4]
        FPS_course_idx = furthest_point_sample(points.transpose(1, 2).contiguous(), self.course) # [b,3,512]

        ROI_anchors1 = self.ROI1_block(ROI_anchors1_idx.permute(0,2,1))
        ROI_anchors2 = self.ROI2_block(ROI_anchors2_idx.permute(0,2,1))
        FPS_course = self.FPS_course(gather_points(points,FPS_course_idx))

        concat_branches = torch.concat([ROI_anchors1,FPS_course,ROI_anchors2], dim=2)
        x_g, coarse = self.b(concat_branches)#.permute(0,2,1))

        fine, feat_fine = self.refine(None, coarse, x_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, x_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:

            loss3 = calc_cd(fine1.contiguous(), gt.contiguous(),norm_=1)
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()
            
            loss2 = calc_cd(fine.contiguous(), gt_fine1.contiguous(),norm_=1)
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

            loss1 = calc_cd(coarse.contiguous(), gt_coarse.contiguous(),norm_=1)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            #return fine1, loss2, total_train_loss
            return total_train_loss
        else:

            cd_p = calc_cd(fine1, gt,norm_=2)
            cd_t = calc_cd(fine1, gt,norm_=1)
            cd_p_coarse = calc_cd(coarse, gt,norm_=2)
            cd_t_coarse = calc_cd(coarse, gt,norm_=1)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}

        #gather_points(x0, FPS_course_idx)


if __name__ == '__main__':
    x = (torch.rand(1, 3, 4096) * 2) - 1
    gt = (torch.rand(1, 16384,3) * 2) - 1

    device = 'cuda'

    x=x.to(device)
    gt=gt.to(device)

    model = Model().to(device=device)

    loss = model(x,gt)

    print(f'This is loss: {loss}')
    print("Parameter count(M):", sum(p.numel() for p in model.parameters())/1_000_000)

    # For now 2025.12.04: parameter count is 21.138971/36.662793 -> 57.6578 % of PointAttn
    # For now 2025.12.05: parameter count is 25.204097/36.6 -> 68.86365 % of PointAttn