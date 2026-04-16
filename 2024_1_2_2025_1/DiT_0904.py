
import torch
import torch.nn as nn

def cosine_beta_schedule(beta_start,beta_end,num_timesteps, s=0.002):
  def f(t):
    return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
  x = torch.linspace(0, num_timesteps, num_timesteps + 1)
  alphas_cumprod = f(x) / f(torch.tensor([0]))
  betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  betas = torch.clip(betas, beta_start,beta_end)
  return betas


class DiffusionModel(nn.Module):
    def __init__(self, time_steps, 
                 beta_start = 1e-4, 
                 beta_end = 0.02,
                 image_dims = (3, 128*2, 128*2)):
        
        super().__init__()
        self.time_steps = time_steps
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        #self.betas = torch.linspace(beta_start, beta_end, self.time_steps) # I have change it to cosine instead of linear...
        self.betas = cosine_beta_schedule(beta_start,beta_end,self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)
        self.model = DiT(256, 24, patch_size=4, 
                 hidden_size=128, num_features=128, 
                 num_layers=6, num_heads=4)

    def add_noise(self, x, ts):
        # 'x' and 'ts' are expected to be batched
        noise = torch.randn_like(x)
        # print(x.shape, noise.shape)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x,y, t):
        return self.model(x,y,t)


def extract_patches(image_tensor, patch_size=8):
    """
    Extracts patches from an image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor with shape (bs, c, h, w).
        patch_size (int, optional): Size of the patches to extract. Defaults to 8.

    Returns:
        torch.Tensor: Extracted patches with shape (bs, L, c * patch_size * patch_size),
                      where L is the number of patches.
    """
    bs, c, h, w = image_tensor.size()

    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)

    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    return unfolded

def reconstruct_image(patch_sequence, image_shape, patch_size=8):
    """
    Reconstructs the original image tensor from a sequence of patches.

    Args:
        patch_sequence (torch.Tensor): Sequence of patches with shape
                                       (bs, L, c * patch_size * patch_size).
        image_shape (tuple): Shape of the original image tensor (bs, c, h, w).
        patch_size (int, optional): Size of the patches used in extraction. Defaults to 8.

    Returns:
        torch.Tensor: Reconstructed image tensor with shape (bs, c, h, w).
    """
    bs, c, h, w = image_shape
    c = c//2
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    unfolded_shape = (bs, num_patches_h, num_patches_w, patch_size, patch_size, c)
    patch_sequence = patch_sequence.view(*unfolded_shape)
    
    patch_sequence = patch_sequence.permute(0, 5, 1, 3, 2, 4).contiguous()
    
    reconstructed = patch_sequence.view(bs, c, h, w)
    
    return reconstructed


class ConditionalNorm2d(nn.Module):
    """
    Conditional Layer Normalization module for 2D inputs.

    This module applies layer normalization and then scales and shifts the normalized
    input based on input features.

    Args:
        hidden_size (int): The size of the hidden dimension to normalize.
        num_features (int): The number of input features for condition.

    Attributes:
        norm (nn.LayerNorm): Layer normalization module.
        fcw (nn.Linear): Linear layer for generating the scaling factor.
        fcb (nn.Linear): Linear layer for generating the shift factor.
    """

    def __init__(self, hidden_size, num_features):
        super(ConditionalNorm2d, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.fcw = nn.Linear(num_features, hidden_size)
        self.fcb = nn.Linear(num_features, hidden_size)

    def forward(self, x, features):
        """
        Forward pass of the ConditionalNorm2d module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            features (torch.Tensor): Conditioning features of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Normalized and conditioned output tensor of the same shape as input x.
        """
        bs, s, l = x.shape
        
        out = self.norm(x)
        w = self.fcw(features).reshape(bs, 1, -1)
        b = self.fcb(features).reshape(bs, 1, -1)

        return w * out + b
    

class SinusoidalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2  # Half sine, half cosine
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-5 * step.unsqueeze(0))  # Avoids instability
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and conditional normalization.

    Args:
        hidden_size (int): Size of the hidden dimension. Default is 128.
        num_heads (int): Number of attention heads. Default is 4.
        num_features (int): Number of features for conditional normalization. Default is 128.

    Attributes:
        norm (nn.LayerNorm): Layer normalization for input.
        multihead_attn (nn.MultiheadAttention): Multi-head attention mechanism.
        con_norm (ConditionalNorm2d): Conditional normalization layer.
        mlp (nn.Sequential): Multi-layer perceptron for feature processing.
    """

    def __init__(self, hidden_size=128, num_heads=4, num_features=128):
        super(TransformerBlock, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                                    batch_first=True, dropout=0.0)
        self.con_norm = ConditionalNorm2d(hidden_size, num_features)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
                
    def forward(self, x, features):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            features (torch.Tensor): Conditional features for normalization.

        Returns:
            torch.Tensor: Processed tensor after attention and MLP layers.
        """
        norm_x = self.norm(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.con_norm(x, features)
        x = self.mlp(norm_x) + x
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) module for vision encoding.

    Args:
        image_size (int): Size of the input image (assuming square images).
        channels_in (int): Number of input channels.
        patch_size (int): Size of image patches. Default is 16.
        hidden_size (int): Size of the hidden dimension. Default is 128.
        num_features (int): Number of features for time embedding. Default is 128.
        num_layers (int): Number of transformer layers. Default is 3.
        num_heads (int): Number of attention heads in each transformer block. Default is 4.

    Attributes:
        time_mlp (nn.Sequential): MLP for time step embedding.
        patch_size (int): Size of image patches.
        fc_in (nn.Linear): Linear layer for patch embedding.
        pos_embedding (nn.Parameter): Learnable positional embeddings.
        blocks (nn.ModuleList): List of TransformerBlock modules.
        fc_out (nn.Linear): Linear layer for output projection.
    """

    def __init__(self, image_size, channels_in, patch_size=16, 
                 hidden_size=128, num_features=128, 
                 num_layers=3, num_heads=4):
        super(DiT, self).__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalEncoding(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU()
        )
        
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
        self.view_embedding = nn.Embedding(4, hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_size, channels_in//2 * patch_size * patch_size)
        
    '''

    def forward(self, image_in, index):
        
        """
        Forward pass of the DiT module.

        Args:
            image_in (torch.Tensor): Input image tensor.
            index (torch.Tensor): Time step index tensor.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        index_features = self.time_mlp(index)

        patch_seq = extract_patches(image_in, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        embs = patch_emb + self.pos_embedding
        
        for block in self.blocks:
            embs = block(embs, index_features)
        
        image_out = self.fc_out(embs)
        
        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size)

    '''
    
    def forward(self, image_in, index):
        bs, c, h, w = image_in.shape
        index_features = self.time_mlp(index)

        patch_seq = extract_patches(image_in, patch_size=self.patch_size)  # (bs, L, c*ps^2)
        patch_emb = self.fc_in(patch_seq)  # (bs, L, hidden_size)

        # --- Assign view ids for each patch ---
        # Channels are grouped as [view1:6, view2:6, ...]
        # Each patch "inherits" a view ID based on which channels contributed.
        num_channels_per_patch = 6 * (self.patch_size ** 2)
        num_patches = patch_emb.size(1)

        # Create view ids (repeat for all patches in sequence)
        view_ids = torch.arange(4, device=image_in.device).repeat_interleave(num_patches // 4)
        view_ids = view_ids.unsqueeze(0).expand(bs, -1)  # (bs, L)

        # Lookup embeddings
        view_embs = self.view_embedding(view_ids)  # (bs, L, hidden_size)

        # --- Add embeddings ---
        embs = patch_emb + self.pos_embedding + view_embs

        for block in self.blocks:
            embs = block(embs, index_features)

        image_out = self.fc_out(embs)
        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size)


# ---- MAIN SCRIPT WITH DUMMIES ----
def main():
    # Dummy settings
    batch_size = 8
    num_views = 4
    channels_per_view = 6
    image_size = 256   # must be divisible by patch_size
    patch_size = 4

    # Create dummy input: (bs, num_views, channels, H, W)
    dummy_input = torch.randn(batch_size, num_views, channels_per_view, image_size, image_size)

    # Reshape to (bs, total_channels, H, W)
    dummy_input = dummy_input.view(batch_size, num_views * channels_per_view, image_size, image_size)

    # Dummy timestep index (like noise level)
    dummy_index = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.float)

    # Create model
    model = DiT(256, 24, patch_size=4, 
                 hidden_size=128, num_features=128, 
                 num_layers=6, num_heads=4)

    # Forward pass
    output = model(dummy_input, dummy_index)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
