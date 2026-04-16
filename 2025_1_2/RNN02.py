import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utility: naive kNN (CPU-safe, slow but simple)
# -------------------------

from pytorch3d.loss import chamfer_distance
def calc_cd(pc1,pc2,norm_=2):
    assert pc1.dim() == 3 and pc2.dim() == 3 and pc1.size(2) == 3 and pc2.size(2) == 3, f"Shape mismatch at calc_cd() -> {pc1.shape} and {pc2.shape}"
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)
    return loss


def knn_points(points, center, k):
    """
    points: [B, N, 3]
    center: [B, 3]
    returns idx: [B, k]
    """
    B, N, _ = points.shape
    center = center.unsqueeze(1)               # [B, 1, 3]
    dist = torch.norm(points - center, dim=-1) # [B, N]
    idx = dist.topk(k, largest=False)[1]
    return idx


def gather_points(points, idx):
    """
    points: [B, N, 3]
    idx: [B, k]
    returns: [B, k, 3]
    """
    B, k = idx.shape
    idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(points, 1, idx)


# -------------------------
# PointNet Encoder
# -------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x):  # [B, N, 3]
        f = self.mlp(x)
        global_feat = torch.max(f, dim=1)[0]
        return global_feat


# -------------------------
# Attention Module
# -------------------------
class AttentionModule(nn.Module):
    def __init__(self, point_feat_dim=3, global_dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + global_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, points, global_feat, hidden):
        """
        points: [B, N, 3]
        global_feat: [B, G]
        hidden: [B, H]
        """
        B, N, _ = points.shape
        g = global_feat.unsqueeze(1).expand(-1, N, -1)
        h = hidden.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([points, g, h], dim=-1)
        scores = self.mlp(x).squeeze(-1)
        alpha = F.softmax(scores, dim=1)
        center = torch.sum(alpha.unsqueeze(-1) * points, dim=1)
        return center, alpha


# -------------------------
# Patch Refinement Network
# -------------------------
class PatchRefiner(nn.Module):
    def __init__(self, hidden_dim=128, k=32, new_points=16):
        super().__init__()
        self.k = k
        self.new_points = new_points

        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k * 3)
        )

        self.new_point_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, new_points * 3)
        )

    def forward(self, hidden):
        B = hidden.shape[0]
        offsets = self.offset_mlp(hidden).view(B, self.k, 3) * 0.01
        new_pts = self.new_point_mlp(hidden).view(B, self.new_points, 3) * 0.02
        return offsets, new_pts

class PatchRefiner(nn.Module):
    def __init__(self, hidden_dim=128, new_points=16):
        super().__init__()
        self.new_points = new_points

        # Only predict new points
        self.new_point_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, new_points * 3)
        )

    def forward(self, hidden):
        B = hidden.shape[0]
        new_pts = self.new_point_mlp(hidden).view(B, self.new_points, 3) * 0.02
        return new_pts

# -------------------------
# Main Recurrent Completion Model
# -------------------------
class Model(nn.Module):
    def __init__(
        self,
        steps=10,
        k=64,
        new_points=1152,
        global_dim=256,
        hidden_dim=128
    ):
        super().__init__()

        self.steps = steps
        self.k = k
        self.new_points = new_points

        self.encoder = PointNetEncoder(global_dim)

        self.attention = AttentionModule(
            point_feat_dim=3,
            global_dim=global_dim,
            hidden_dim=hidden_dim
        )

        # LSTM controller
        self.lstm = nn.LSTMCell(
            input_size=global_dim + 3 + 3,
            hidden_size=hidden_dim
        )

        self.refiner = PatchRefiner(
            hidden_dim=hidden_dim,

            new_points=new_points
        )

    def loss(self,y_hat,gt):
        return calc_cd(y_hat.contiguous(), gt.contiguous(),norm_=2).mean()
    def forward(self, points, gt, is_training=True):
        """
        points: [B, 4096, 3]
        gt:     [B, N_gt, 3]
        """
        B, N0, _ = points.shape
        device = points.device

        # LSTM state
        h = torch.zeros(B, 128, device=device)
        c = torch.zeros(B, 128, device=device)

        canvas = points

        # (optional) store intermediate canvases for later supervision
        # canvases = []

        for t in range(self.steps):
            # Global encoding
            global_feat = self.encoder(canvas)

            # Attention selects refinement center
            center, _ = self.attention(canvas, global_feat, h)

            # Local patch (we can still compute patch feature if needed)
            idx = knn_points(canvas, center, self.k)
            patch = gather_points(canvas, idx)
            patch_feat = (patch - center.unsqueeze(1)).mean(dim=1)

            # LSTM step
            lstm_input = torch.cat([global_feat, center, patch_feat], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            # Generate new points only
            new_pts = self.refiner(h)
            new_pts = new_pts + center.unsqueeze(1)

            # Update canvas (add only new points)
            canvas = torch.cat([canvas, new_pts], dim=1)

        # ---------------------------
        # LOSS: split input vs new
        # ---------------------------
        P_input = canvas[:, :N0, :]   # original points
        P_new   = canvas[:, N0:, :]   # predicted points

        cd_input = self.loss(P_input, gt)
        cd_new   = self.loss(P_new,   gt)

        # weight anchors lightly, predictions strongly
        total_loss = 0.1 * cd_input + 1.0 * cd_new

        if is_training:
            return total_loss
        else:
            return canvas, total_loss
     

# -------------------------
# Parameter Counter
# -------------------------
def count_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:40s} {p.numel():8d}")
            total += p.numel()
    print(f"\nTotal trainable parameters: {total:,}")


# -------------------------
# Dummy main()
# -------------------------
def main():
    torch.manual_seed(0)

    B = 2
    N = 2048*2

    dummy_input = torch.randn(B, N, 3)

    model = Model()

    print("\n=== RUNNING FORWARD PASS ===")
    output = model(dummy_input,dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
