import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention

from einops import rearrange
from ViT_Preprocess import VisionTransformer

# cos schedule from here:
# https://dzdata.medium.com/intro-to-diffusion-model-part-4-62bd94bd93fd
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
        self.model = MultiViewUNet_changed(input_channels = c, output_channels = c, time_steps = self.time_steps)

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

# ------------------------------------------------ From here I'll write the MultiViewUNet  ---------------------------------

class CrossViewAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels,channels),
        )
    def forward(self, x, type = "",n_views=4):
        """
        x: Tensor of shape [B, V, C, H, W]
        Returns: same shape tensor, but views mixed via attention
        """
        if type == "downsample":
            B, V, C, H, W = x.shape
            x = x.permute(0, 3, 4, 1, 2)      # [B, H, W, V, C]
            #x = x.reshape(B * H * W, V, C)    # [B*H*W, V, C]
            x = x.reshape(B, H * W *V, C)
            # B, H*w*c,V
        else:
            B, C, H, W = x.shape
            V = n_views
            C = C//V
            x = x.permute(0, 2, 3, 1)      # [B, H, W, C]
            #x = x.reshape(B * H * W, V, C)    # [B*H*W, V, C]
            x = x.reshape(B, H * W * V, C)

        # Apply self-attention across the views
        x_ln = self.norm(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)  # [B*H*W, V, C]
        attn_out = attn_out + x
        attn_out = self.ff(attn_out) + attn_out
        x = attn_out.view(B, H, W, V, C).permute(0, 3, 4, 1, 2)  # back to [B, V, C, H, W]
        if type == "downsample":
            
            x = x.reshape(B, V * C, H, W) # added later...
        return x

      
 
# ------------------------------------- From here I will show the changed version 2025.05. --------------------------

# Double Conv Block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps = 250, activation = "", embedding_dims = None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        #self.bn1 = nn.GroupNorm(1,out_c)
        #self.bn1 = nn.GroupNorm(num_groups=32, num_channels=out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

        #self.bn2 = nn.GroupNorm(1,out_c)
        #self.bn2 = nn.GroupNorm(num_groups=32, num_channels=out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.embedding_dims = embedding_dims if embedding_dims else out_c
        
        self.embedding = SinusoidalEncoding(self.embedding_dims)
        #self.embedding = nn.Embedding(num_embeddings=time_steps, embedding_dim=self.embedding_dims)
        # switch to nn.Embedding if you want to pass in timestep instead; but note that it should be of dtype torch.long
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU() if activation == "Relu" else nn.SiLU()
        
    def forward(self, inputs, time = None):
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = x + time_embedding
        return x

class AttnBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.ln = nn.LayerNorm(embedding_dims)
        self.mhsa = MultiHeadSelfAttention(embedding_dims = embedding_dims, num_heads = num_heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.GELU(),
            nn.Linear(self.embedding_dims, self.embedding_dims),
        )
 
    def forward(self, x):
        bs, c, sz, _ = x.shape
        x = x.view(-1, self.embedding_dims, sz * sz).swapaxes(1, 2) # is of the shape (bs, sz**2, self.embedding_dims)
        x_ln = self.ln(x)
        _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, c, sz, sz)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
        self.head_dim = self.embedding_dims // self.num_heads
        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)
        self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)

    def attention(self, q, k, v):
        # no need for a mask
        attn_weights = F.softmax((q @ k.transpose(-1, -2))/self.head_dim**0.5, dim = -1)
        return attn_weights, attn_weights @ v        

    def forward(self, q, k, v):
        bs, img_sz, c = q.shape
        q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v of the shape (bs, self.num_heads, img_sz**2, self.head_dim)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        attn_weights, o = self.attention(q, k, v) # of shape (bs, num_heads, img_sz**2, c)
        
        o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
        o = self.wo(o)
        return attn_weights, o


# Encoder Block for downsampling
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()
        self.conv = conv_block(in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time = None):
        x = self.conv(inputs, time)
        p = self.pool(x)
        return x, p

# Decoder Block for upsampling
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)
    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)
        return x

class CrossViewAttention2(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels,channels),
        )
    def forward(self, x, type = "",n_views=4):
        """
        x: Tensor of shape [B, V, C, H, W]
        Returns: same shape tensor, but views mixed via attention
        """
        if type == "downsample" or type == "skip":
            B, V, C, H, W = x.shape
            x = x.permute(0, 3, 4, 1, 2)      # [B, H, W, V, C]
            #x = x.reshape(B * H * W, V, C)    # [B*H*W, V, C]
            x = x.reshape(B, H * W *V, C)
            # B, H*w*c,V
        else:
            B, C, H, W = x.shape
            V = n_views
            C = C//V
            x = x.permute(0, 2, 3, 1)      # [B, H, W, C]
            #x = x.reshape(B * H * W, V, C)    # [B*H*W, V, C]
            x = x.reshape(B, H * W * V, C)

        # Apply self-attention across the views
        x_ln = self.norm(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)  # [B*H*W, V, C]
        attn_out = attn_out + x
        attn_out = self.ff(attn_out) + attn_out
        x = attn_out.view(B, H, W, V, C).permute(0, 3, 4, 1, 2)  # back to [B, V, C, H, W]
        if type == "downsample":
            x = x.reshape(B, V * C, H, W) # added later...
        return x

class MultiViewUNet_changed(nn.Module):
    def __init__(self, input_channels = 6, output_channels = 3, time_steps = 512, n_views = 4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.n_views = n_views

        #self.ViT_Preprocess = ViT_Preprocess()

        self.start_dim = 32
        self.e1 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps) 
        self.e2 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps)
        #self.da2 = AttnBlock(self.start_dim*2)
        self.e3 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps)
        self.da3 = AttnBlock(self.start_dim*4)

        self.cross_view_attention = CrossViewAttention2(self.start_dim*4)
        
        # From here, I have used a concatenated fusion of the 4 views...
        self.e4 = encoder_block(self.start_dim*4*self.n_views, self.start_dim*8*self.n_views, time_steps=self.time_steps)
        self.b = conv_block(self.start_dim*8*self.n_views, self.start_dim*16*self.n_views, time_steps=self.time_steps) # bottleneck

        self.d1 = decoder_block(self.start_dim*16*self.n_views, self.start_dim*8*self.n_views, time_steps=self.time_steps)

        self.cross_view_attention_upsample = CrossViewAttention2(self.start_dim*8)
        self.d2 = decoder_block(self.start_dim*8, self.start_dim*4, time_steps=self.time_steps) 
        self.d3 = decoder_block(self.start_dim*4, self.start_dim*2, time_steps=self.time_steps)
        #self.attn_decoder = AttnBlock(self.start_dim*2)
        self.d4 = decoder_block(self.start_dim*2, self.start_dim, time_steps=self.time_steps)        
        self.outputs = nn.Conv2d(self.start_dim, self.output_channels, kernel_size=1, padding=0)

    def forward(self,inputs_sparse, input_noised,t):
        B,V,C,H,W = inputs_sparse.shape

        #rearrange_sparse = rearrange(inputs_sparse, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
        #ViT_output = self.ViT_Preprocess(inputs_sparse)
        #ViT_output_recovered = rearrange(ViT_output, 'b c (b1 h) (b2 w) -> b (b1 b2) c h w', b1=2, b2=2)
        #attn_noised = self.NoisedAttn(input_noised)
        
        inputs = torch.cat([inputs_sparse, input_noised], dim = 2) # inputs_sparse is changed!!!!!!!!!!!!
        features = []
#        skip_connections = []
        skip_connections = [[], [], []]  # for s1s, s2s, s3s

        #Encode per-view:
        for view in range(V):
            temp = inputs[:, view]  # (B, 6, H, W)

            s1, p1 = self.e1(temp, t)

            s2, p2 = self.e2(p1, t)
            #p2 = self.da2(p2)
            s3, p3 = self.e3(p2, t)
            p3 = self.da3(p3)

            skip_connections[0].append(s1)  
            skip_connections[1].append(s2)  
            skip_connections[2].append(s3)
            features.append(p3) # might delete only bottleneck...

        fused = torch.stack(features,dim=1)
        attn_fused = self.cross_view_attention(fused,"downsample")
        s4, p4 = self.e4(attn_fused, t)
        
        b = self.b(p4, t)
        
        d1 = self.d1(b, s4, t)
        attn_fused_upsample = self.cross_view_attention_upsample(d1,"upsample")

        outputs = []
        for view in range(V):
            temp = attn_fused_upsample[:, view]  # (B, C, H, W)
            d2 = self.d2(temp, skip_connections[2][view], t)
            
            d3 = self.d3(d2, skip_connections[1][view], t)
            d4 = self.d4(d3, skip_connections[0][view], t)

            d4 = self.outputs(d4)
            outputs.append(d4)

        outputs = torch.stack(outputs,dim =1)
        
        return outputs


##### 07.19 changes to the attention and cnn part............................................................................................................................................................
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps=250, activation="", kernel_size=3, stride=1, embedding_dims=None):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, stride=stride, padding=self.kernel_size // 2)
        #self.bn1 = nn.GroupNorm(num_groups=4, num_channels=out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        #self.bn2 = nn.GroupNorm(num_groups=4, num_channels=out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.embedding_dims = embedding_dims if embedding_dims else out_c
        self.embedding = SinusoidalEncoding(self.embedding_dims)
        
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU() if activation == "Relu" else nn.SiLU()

    def forward(self, inputs, time=None):
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = x + time_embedding
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation="relu", kernel_size=3, use_pooling=False,return_skip = False):
        super().__init__()
        self.return_skip = return_skip
        stride = 1 if use_pooling else 2
        self.conv = conv_block(in_c, out_c, time_steps=time_steps, activation=activation, kernel_size=kernel_size, stride=stride, embedding_dims=out_c)
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time=None):
        x = self.conv(inputs, time)
        if self.use_pooling:
            p = self.pool(x)
        else:
            p = x  # already downsampled by stride in conv
        if self.return_skip:
            return x, p
        else:
            return p

class decoder_block(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_steps, activation="relu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + skip_c, out_c, time_steps=time_steps, activation=activation, embedding_dims=out_c)

    def forward(self, inputs, skip, time=None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B x embed_dim x H/P x W/P
        x = x.flatten(2).transpose(1, 2)  # B x N x embed_dim
        return x




# ---------------------------------- changed 0724 -------------------------------------------------
import math


class MultiViewUNet_changed(nn.Module):
    def __init__(self, input_channels = 6, output_channels = 3, time_steps = 512, n_views = 4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.n_views = n_views

        self.start_dim = self.input_channels*2
        self.bottleneck_xy = 32

        # First encode layer (sum the given 3,5,7 convs):
        self.e1_3 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps, kernel_size=3, use_pooling=True)
        self.e1_5 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps,kernel_size=5) 
        self.e1_7 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps,kernel_size=7) 
        # 2 * 6 = 12
        self.level_1_view_encoder_layer = nn.TransformerEncoderLayer(d_model=self.start_dim*3, nhead=2)
        self.level_1_view_transformer = nn.TransformerEncoder(self.level_1_view_encoder_layer, num_layers=2)
        self.head_1 = nn.Linear(self.start_dim*3, self.start_dim)#patch_size * patch_size * 64)
        self.register_buffer("pos_view_1", self.get_sinusoidal_positional_encoding(128*128, self.start_dim*3))

        # Second:
        self.e2_3 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=3, use_pooling=True,return_skip=False)
        self.e2_5 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=5)
        self.e2_7 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=7)
        # 2 * 12 = 24
        self.level_2_view_encoder_layer = nn.TransformerEncoderLayer(d_model=self.start_dim*2*3, nhead=2)
        self.level_2_view_transformer = nn.TransformerEncoder(self.level_2_view_encoder_layer, num_layers=2)
        self.head_2 = nn.Linear(self.start_dim*2*3, self.start_dim*2) #patch_size * patch_size * 64)
        self.register_buffer("pos_view_2", self.get_sinusoidal_positional_encoding(64*64, self.start_dim*2*3))
        
        # Third:
        self.e3_3 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,  kernel_size=3, use_pooling=True,return_skip=False)
        self.e3_5 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,kernel_size=5)
        self.e3_7 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,kernel_size=7)
        # 2 * 24 = 48
        self.level_3_view_encoder_layer = nn.TransformerEncoderLayer(d_model=self.start_dim*4*3, nhead=2)
        self.level_3_view_transformer = nn.TransformerEncoder(self.level_3_view_encoder_layer, num_layers=2)
        self.head_3 = nn.Linear(self.start_dim*4*3, self.start_dim*4)#patch_size * patch_size * 64)
        self.register_buffer("pos_view_3", self.get_sinusoidal_positional_encoding(32*32, self.start_dim*4*3))
        # Forth:
        #self.e4_3 = encoder_block(self.start_dim*4, self.start_dim*8, time_steps=self.time_steps,  kernel_size=3, use_pooling=True,return_skip=True)
        #self.e4_5 = encoder_block(self.start_dim*4, self.start_dim*8, time_steps=self.time_steps,kernel_size=5)
        #self.e4_7 = encoder_block(self.start_dim*4, self.start_dim*8, time_steps=self.time_steps,kernel_size=7)
        # 2 * 48 = 96
        # Same transformer per view as at bottleneck but its per view before the global, concatenating all the branches (3x3,5x5 and 7x7)

        # At the end of the loop:
        #self.view_encoder_layer = nn.TransformerEncoderLayer(d_model=self.start_dim*4*3, nhead=2)
        #self.view_transformer = nn.TransformerEncoder(self.view_encoder_layer, num_layers=4)
        #self.head = nn.Linear(self.embed_dim, 4 * 3*64)#patch_size * patch_size * 64)

        ## Bottleneck:
        # Here the data should be dealt with: 
        # we have 4x3x64x(16x16) input. create a transformer with depth 4 heads 4. One path is embeded from the inputs we
        # gonna have 3x16x16 patches, each patch with a depth of 4x64(embeded alter and added positional embeddin, learnt).
        # the output or the head of the transformer should be 4x64x16x16, so each view will have its global learnt thingy that ca
        # be added later with the skip connections. This aprt is for sharing infomration...
        self.embed_dim = 128
        heads = 4
        depth = 4
        patch_size = 16

        transformer_encoderlayer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(transformer_encoderlayer, num_layers=depth)
        self.head = nn.Linear(self.embed_dim, self.start_dim*4*4)#patch_size * patch_size * 64)
        
        #self.patch_embed = PatchEmbed(in_chans=4*3*64, patch_size=1, embed_dim=self.embed_dim)
        
        self.patch_embed = nn.Linear(4 * (self.start_dim*4), self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.bottleneck_xy*self.bottleneck_xy, self.embed_dim))
        
        ## Decode:
        # Than skip connection of 3x3 added also to upsampling at decoding phase
        # Every view will use the global part from the bottleneck that is for its current use (hence the 4x64x16x16)
        #self.d1 = decoder_block(192,64, self.start_dim*4, time_steps=self.time_steps)
        self.d2 = decoder_block(self.start_dim*4,self.start_dim*4, self.start_dim*4, time_steps=self.time_steps) 
        self.d3 = decoder_block(self.start_dim*4,self.start_dim*2, self.start_dim*2, time_steps=self.time_steps) 
        self.d4 = decoder_block(self.start_dim*2,self.start_dim, self.start_dim, time_steps=self.time_steps)        
        
        self.outputs = nn.Conv2d(self.start_dim, self.output_channels, kernel_size=1, padding=0)

        # For fixed sinusoidal embedding:
        seq_len = 32 * 32  # for 16x16 feature maps
        self.register_buffer("pos_embed_view", self.get_sinusoidal_positional_encoding(seq_len, self.start_dim*4*3))
        self.register_buffer("pos_embed_global", self.get_sinusoidal_positional_encoding(seq_len, self.embed_dim))

    def forward(self,inputs_sparse, input_noised,t):
        B,V,C,H,W = inputs_sparse.shape
        inputs = torch.cat([inputs_sparse, input_noised], dim = 2) 
        features = []
        skip_connections = [[], [], [], []]  # for s1s, s2s, s3s

        #Encode per-view:
        for view in range(V):
            temp = inputs[:, view]  # (B, 6, H, W)

            # layer 1:
            l1_3 = self.e1_3(temp, t)
            l1_5 = self.e1_5(temp, t)
            l1_7 = self.e1_7(temp, t)
            s1 = self.encoder_transformers(l1_3,l1_5,l1_7)
            # layer 2:
            l2_3 = self.e2_3(s1, t)
            l2_5 = self.e2_5(s1, t)
            l2_7 = self.e2_7(s1, t)
            s2 = self.encoder_transformers(l2_3,l2_5,l2_7)
            
            # layer 3:
            l3_3 = self.e3_3(s2, t)
            l3_5 = self.e3_5(s2, t)
            l3_7 = self.e3_7(s2, t)
            s3 = self.encoder_transformers(l3_3,l3_5,l3_7)

            # Saving skip connections for later:            
            skip_connections[0].append(l1_5)  
            skip_connections[1].append(l2_5)  
            skip_connections[2].append(l3_5)

            features.append(s3)

        features = torch.stack(features, dim=1)

        B, V, C, H, W = features.shape
        patches = features.permute(0, 3, 4, 1, 2)  # [B, H, W, V, C]
        patches = patches.reshape(B, H * W, V * C)  # [B, N_patches, D_embed]

        #fused = torch.stack(features,dim=1)
        
        # Bottleneck:
        # Here the transformer should have the input fused,
        # Here I need each patch to be created and have a positional encoded (learnt),
        # also the output should of the transformer should have 4x64x16x16 that is
        # fused_out so at the decoder as shwon i can use it per view to upsample with skip conenction of 3x3
        fused = self.patch_embed(patches)  
        fused = fused + self.pos_embed_global
                
        fused = fused.transpose(0, 1)  # [H*W, B, embed_dim]
        fused = self.transformer(fused)  # same shape
        fused = fused.transpose(0, 1)  # [B, H*W, embed_dim]  

        fused = self.head(fused)  # [B, H*W, V*C]
        fused = fused.view(B, H, W, V, C)
        fused = fused.permute(0, 3, 4, 1, 2)  # [B, V, C, H, W]
        
        outputs = []
        for view in range(V):
            temp = fused[:, view]  # (B, C, H, W)
            #d1 = self.d1(temp, skip_connections[3][view], t)
            d2 = self.d2(temp, skip_connections[2][view], t)
            d3 = self.d3(d2, skip_connections[1][view], t)
            
            d4 = self.d4(d3, skip_connections[0][view], t)

            d4 = self.outputs(d4)
            outputs.append(d4)

        outputs = torch.stack(outputs,dim =1)
        
        return outputs

    def get_sinusoidal_positional_encoding(self, seq_len, dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, dim]

    def encoder_transformers(self, s3, s5, s7):
        # Combine multi-kernel features along channel dimension
        combined = torch.cat([s3, s5, s7], dim=1)  # [B, C_total, H, W]
        B, C, H, W = combined.shape

        # Flatten spatial dims -> sequence for transformer
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Choose transformer & head based on channel count
        if C == self.start_dim * 3:
            transformer = self.level_1_view_transformer
            head = self.head_1
            pos_embed = self.pos_view_1
        elif C == self.start_dim * 2 * 3:
            transformer = self.level_2_view_transformer
            head = self.head_2
            pos_embed = self.pos_view_2
        elif C == self.start_dim * 4 * 3:
            transformer = self.level_3_view_transformer
            head = self.head_3
            pos_embed = self.pos_view_3
        else:
            raise ValueError(f"Unexpected channel count in encoder_transformers: {C}")

        # Add positional encoding
        patches = patches + pos_embed[:, :H * W, :]

        # Transformer expects [sequence_len, batch, embedding_dim]
        patches = patches.transpose(0, 1)
        transformed = transformer(patches)
        transformed = transformed.transpose(0, 1)

        # Project back to the reduced channel size
        transformed = head(transformed)  # [B, H*W, C_reduced]
        transformed = transformed.view(B, H, W, -1).permute(0, 3, 1, 2)

        return transformed


import torch
import torch.nn as nn
import math

class MultiViewUNet_changed(nn.Module):
    def __init__(self, input_channels = 6, output_channels = 3, time_steps = 512, n_views = 4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.n_views = n_views

        self.start_dim = self.input_channels*2
        self.bottleneck_xy = 32

        self.inception_fuse_1 = nn.Conv2d(
            in_channels=self.start_dim * 3,
            out_channels=self.start_dim,  # match the expected next stage input
            kernel_size=1
        )


        # ===== Level 1 =====
        self.e1_3 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps, kernel_size=3, use_pooling=True,return_skip=True)
        self.e1_5 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps,kernel_size=5) 
        self.e1_7 = encoder_block(self.input_channels, self.start_dim, time_steps=self.time_steps,kernel_size=7) 

        self.level_1_patch = 8
        level1_d_model = self.start_dim*3*(self.level_1_patch**2)
        self.level_1_view_encoder_layer = nn.TransformerEncoderLayer(d_model=level1_d_model, nhead=2)
        self.level_1_view_transformer = nn.TransformerEncoder(self.level_1_view_encoder_layer, num_layers=2)
        
        self.head_1 = nn.Linear(level1_d_model, self.start_dim * (self.level_1_patch**2))
        self.register_buffer("pos_view_1", self.get_sinusoidal_positional_encoding(
            (128//self.level_1_patch)*(128//self.level_1_patch), 
            level1_d_model
        ))

        # ===== Level 2 =====
        self.e2_3 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=3, use_pooling=True,return_skip=True)
        self.e2_5 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=5)
        self.e2_7 = encoder_block(self.start_dim, self.start_dim*2, time_steps=self.time_steps,kernel_size=7)

        self.level_2_patch = 4
        level2_d_model = self.start_dim*2*3*(self.level_2_patch**2)
        self.level_2_view_encoder_layer = nn.TransformerEncoderLayer(d_model=level2_d_model, nhead=2)
        self.level_2_view_transformer = nn.TransformerEncoder(self.level_2_view_encoder_layer, num_layers=2)
        self.head_2 = nn.Linear(level2_d_model, self.start_dim*2 * (self.level_2_patch**2))
        self.register_buffer("pos_view_2", self.get_sinusoidal_positional_encoding(
            (64//self.level_2_patch)*(64//self.level_2_patch), 
            level2_d_model
        ))
        
        # ===== Level 3 =====
        self.e3_3 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,  kernel_size=3, use_pooling=True,return_skip=True)
        self.e3_5 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,kernel_size=5)
        self.e3_7 = encoder_block(self.start_dim*2, self.start_dim*4, time_steps=self.time_steps,kernel_size=7)

        self.level_3_patch = 2
        level3_d_model = self.start_dim*4*3*(self.level_3_patch**2)
        self.level_3_view_encoder_layer = nn.TransformerEncoderLayer(d_model=level3_d_model, nhead=2)
        self.level_3_view_transformer = nn.TransformerEncoder(self.level_3_view_encoder_layer, num_layers=2)
        self.head_3 = nn.Linear(level3_d_model, self.start_dim*4 * (self.level_3_patch**2))
        self.register_buffer("pos_view_3", self.get_sinusoidal_positional_encoding(
            (32//self.level_3_patch)*(32//self.level_3_patch), 
            level3_d_model
        ))

        ## ===== Bottleneck =====
        self.embed_dim = 128
        heads = 4
        depth = 4

        transformer_encoderlayer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(transformer_encoderlayer, num_layers=depth)
        self.head = nn.Linear(self.embed_dim, self.start_dim*4*4)
        
        self.patch_embed = nn.Linear(4 * (self.start_dim*4), self.embed_dim)
        self.register_buffer("pos_embed_global", self.get_sinusoidal_positional_encoding(
            self.bottleneck_xy*self.bottleneck_xy, self.embed_dim
        ))
        
        ## ===== Decoder =====
        self.d2 = decoder_block(self.start_dim*4,self.start_dim*4, self.start_dim*4, time_steps=self.time_steps) 
        self.d3 = decoder_block(self.start_dim*4,self.start_dim*2, self.start_dim*2, time_steps=self.time_steps) 
        self.d4 = decoder_block(self.start_dim*2,self.start_dim, self.start_dim, time_steps=self.time_steps)        
        
        self.outputs = nn.Conv2d(self.start_dim, self.output_channels, kernel_size=1, padding=0)

    def forward(self,inputs_sparse, input_noised,t):
        B,V,C,H,W = inputs_sparse.shape
        inputs = torch.cat([inputs_sparse, input_noised], dim = 2) 
        features = []
        skip_connections = [[], [], [], []]

        for view in range(V):
            temp = inputs[:, view]  

            l1_3_S,l1_3 = self.e1_3(temp, t)
            l1_5 = self.e1_5(temp, t)
            l1_7 = self.e1_7(temp, t)
            #s1 = self.encoder_transformers(l1_3,l1_5,l1_7, self.level_1_patch)
            l1_concat = torch.cat([l1_3, l1_5, l1_7], dim=1)  # B, 3*start_dim, H, W
            s1 = self.inception_fuse_1(l1_concat)
            
            
            l2_3_S,l2_3 = self.e2_3(s1, t)
            l2_5 = self.e2_5(s1, t)
            l2_7 = self.e2_7(s1, t)
            s2 = self.encoder_transformers(l2_3,l2_5,l2_7, self.level_2_patch)
            
            l3_3_S,l3_3 = self.e3_3(s2, t)
            l3_5 = self.e3_5(s2, t)
            l3_7 = self.e3_7(s2, t)
            s3 = self.encoder_transformers(l3_3,l3_5,l3_7, self.level_3_patch)

            skip_connections[0].append(l1_3_S)  
            skip_connections[1].append(l2_3_S)  
            skip_connections[2].append(l3_3_S)

            features.append(s3)

        features = torch.stack(features, dim=1)

        B, V, C, H, W = features.shape
        patches = features.permute(0, 3, 4, 1, 2)  
        patches = patches.reshape(B, H * W, V * C)  

        fused = self.patch_embed(patches)  
        fused = fused + self.pos_embed_global
        fused = fused.transpose(0, 1)  
        fused = self.transformer(fused)  
        fused = fused.transpose(0, 1)  

        fused = self.head(fused)  
        fused = fused.view(B, H, W, V, C)
        fused = fused.permute(0, 3, 4, 1, 2)  
        
        outputs = []
        for view in range(V):
            temp = fused[:, view]  
            d2 = self.d2(temp, skip_connections[2][view], t)
            d3 = self.d3(d2, skip_connections[1][view], t)
            d4 = self.d4(d3, skip_connections[0][view], t)
            d4 = self.outputs(d4)
            outputs.append(d4)

        outputs = torch.stack(outputs,dim =1)
        return outputs

    def get_sinusoidal_positional_encoding(self, seq_len, dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def patchify(self, x, patch_size):
        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.permute(0,2,3,1,4,5).reshape(B, -1, C*patch_size*patch_size)
        return x

    def unpatchify(self, x, patch_size, H, W):
        B, N, D = x.shape
        C = D // (patch_size*patch_size)
        h = H // patch_size
        w = W // patch_size
        x = x.view(B, h, w, C, patch_size, patch_size)
        x = x.permute(0,3,1,4,2,5).reshape(B, C, H, W)
        return x

    def encoder_transformers(self, s3, s5, s7, patch_size):
        combined = torch.cat([s3, s5, s7], dim=1)
        B, C, H, W = combined.shape

        if C == self.start_dim * 3:
            transformer = self.level_1_view_transformer
            head = self.head_1
            pos_embed = self.pos_view_1
        elif C == self.start_dim * 2 * 3:
            transformer = self.level_2_view_transformer
            head = self.head_2
            pos_embed = self.pos_view_2
        elif C == self.start_dim * 4 * 3:
            transformer = self.level_3_view_transformer
            head = self.head_3
            pos_embed = self.pos_view_3
        else:
            raise ValueError(f"Unexpected channel count in encoder_transformers: {C}")

        patches = self.patchify(combined, patch_size)
        patches = patches + pos_embed[:, :patches.shape[1], :]
        patches = patches.transpose(0, 1)
        transformed = transformer(patches)
        transformed = transformed.transpose(0, 1)
        transformed = head(transformed)
        transformed = self.unpatchify(transformed, patch_size, H, W)
        return transformed


if __name__ == "__main__":
    # Dummy input dimensions
    B = 8       # batch size
    V = 4       # number of views
    C = 3       # channels per input
    H = W = 256 # spatial resolution
    time_steps = 512

    # Create dummy data for sparse and noised inputs
    inputs_sparse = torch.randn(B, V, C, H, W)
    input_noised = torch.randn(B, V, C, H, W)

    # Dummy time tensor (can be random or fixed)
    t = torch.randint(0, time_steps, (B,))

    # Initialize the model
    model = MultiViewUNet_changed()
    # Run forward pass
    outputs = model(inputs_sparse, input_noised, t)

    # Print output shape
    print(f"Output shape: {outputs.shape}")
    # Expected: [B, V, output_channels, H, W] => [8, 4, 3, 256, 256]