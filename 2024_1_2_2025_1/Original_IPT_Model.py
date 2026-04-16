import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDenoiser(nn.Module):
    def __init__(self, img_size=(128, 128), patch_size=4, num_channels=3, 
                 embedding_dim=1024, num_heads=8, num_layers=16, dropout=0.2):
        super(ImageDenoiser, self).__init__()
        
        # Validate img_size compatibility
        print()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            "Image dimensions must be divisible by patch size"
        
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        
        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Enhanced Patch embedding with MLP
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size**2 * num_channels, embedding_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))
        
        # Transformer Encoder with pre-LN and enhanced settings
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LayerNorm for stability
            ),
            num_layers=num_layers
        )
        
        # Enhanced Decoder with MLP
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, patch_size**2 * num_channels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, std=0.02)  # For positional embeddings

    def forward(self, x):
        # Store original input for residual connection
        identity = x
        
        # Extract patches
        B, C, H, W = x.shape
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.permute(0, 2, 1)  # (B, num_patches, patch_dim^2*C)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.encoder(x)
        
        # Decode patches
        x = self.decoder(x)
        
        # Reconstruct image
        x = x.permute(0, 2, 1).contiguous()
        x = F.fold(
            x,
            output_size=self.img_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Add residual connection
        x = x + identity
        
        return x