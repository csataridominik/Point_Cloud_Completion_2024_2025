
#imports

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
from time import time
import gc
from dataset import manage_dataloaders,load_dataloader
from MultiViewDiffusionModel import sparsing, update_loss_png

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTInpainting(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768//4, depth=6, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(3, patch_size, embed_dim)
        #self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(embed_dim, patch_size * patch_size * 3)
        
        with torch.no_grad():
            self.register_buffer('pos_embed', get_sinusoidal_positional_encoding(self.num_patches, embed_dim, device='cpu'))
    def forward(self, x):
        x = self.patch_embed(x)  # B x N x D
        x = x + self.pos_embed
        
        x = self.transformer(x)  # B x N x D
        x = self.head(x)  # B x N x (P*P*3)

        B, N, _ = x.shape
        x = x.view(B, N, 3, self.patch_size, self.patch_size)  # B x N x 3 x P x P
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # B x 3 x N x P x P
        grid_size = int(N ** 0.5)
        x = x.view(B, 3, grid_size, grid_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # B x 3 x H x P x W x P
        x = x.view(B, 3, grid_size * self.patch_size, grid_size * self.patch_size)
        return x


def get_sinusoidal_positional_encoding(n_positions, dim, device):
    position = torch.arange(0, n_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-np.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: (1, n_positions, dim)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B x embed_dim x H/P x W/P
        x = x.flatten(2).transpose(1, 2)  # B x N x embed_dim
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, activation="", embedding_dims=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=out_c, num_channels=out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=out_c, num_channels=out_c)
        self.embedding_dims = embedding_dims if embedding_dims else out_c
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU() if activation.lower() == "relu" else nn.SiLU()

    def forward(self, inputs, time=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation="relu"):
        super().__init__()
        self.conv = conv_block(in_c, out_c, activation=activation, embedding_dims=out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs, time=None):
        x = self.conv(inputs, time)
        p = self.pool(x)
        return x, p
    
class encoder_block_no_pool(nn.Module):
    def __init__(self, in_c, out_c, activation="relu"):
        super().__init__()
        self.conv = conv_block(in_c, out_c, activation=activation, embedding_dims=out_c)
        

    def forward(self, inputs, time=None):
        x = self.conv(inputs, time)
        
        return x, x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation="relu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = conv_block(in_c, out_c, activation=activation)

    def forward(self, x, skip):
        x = self.up(x)
        # concatenate skip connection along channel dim
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ViT_pcc(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768//8, depth=6, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.start_dim = 12
        self.e1 = encoder_block(3, self.start_dim)
        self.e2 = encoder_block(self.start_dim, self.start_dim * 2)
        self.e3 = encoder_block(self.start_dim * 2, self.start_dim * 4)
        self.e4 = encoder_block_no_pool(self.start_dim * 4, self.embed_dim)  # Match embed_dim for transformer input

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(self.embed_dim, patch_size * patch_size * 3)
        self.patch_embed = PatchEmbed(3, self.patch_size, self.embed_dim)
        # Positional embedding on CPU, moved to device in forward
        self.register_buffer('pos_embed', get_sinusoidal_positional_encoding(self.num_patches, embed_dim, device='cpu'))
    
    def forward(self, x):
            """
            x: (B, V, 3, H, W) — RGB = XYZ projections
            Returns:
                out: (B, V, 3, H_out, W_out)
            """
            B, V, C, H, W = x.shape
            features = []

            for v in range(V):
                temp = x[:, v]  # (B, 3, H, W)
                f1, p1 = self.e1(temp)
                f2, p2 = self.e2(p1)
                f3, p3 = self.e3(p2)
                f4, p4 = self.e4(p3)  # (B, embed_dim, H', W')
                features.append(p4)

            cnn_feat = torch.stack(features, dim=1)  # (B, V, C, H', W')

            B, V, C, Hf, Wf = cnn_feat.shape
            cnn_feat = cnn_feat.permute(0, 2, 1, 3, 4).reshape(B, C, V * Hf * Wf)  # (B, C, N_tokens)
            cnn_feat = cnn_feat.permute(0, 2, 1)  # (B, N_tokens, C)

            # Add positional encoding
            pos_embed = self.pos_embed[:, :cnn_feat.size(1)].to(cnn_feat.device)
            tokens = cnn_feat + pos_embed  # (B, N_tokens, C)

            # Transformer expects (N_tokens, B, C), so permute:
            tokens = tokens.permute(1, 0, 2)
            tokens = self.transformer(tokens)
            tokens = tokens.permute(1, 0, 2)  # back to (B, N_tokens, C)

            decoded = self.head(tokens)  # (B, N_tokens, patch_dim)
            patch_dim = self.patch_size

            # reshape to (B, V, Hf, Wf, 3, p, p)
            decoded = decoded.view(B, V, Hf, Wf, 3, patch_dim, patch_dim)
            decoded = decoded.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # (B, V, 3, Hf, p, Wf, p)
            out = decoded.view(B, V, 3, Hf * patch_dim, Wf * patch_dim)

            return out
    
def masked_l1_loss(output, target, mask):
    mask_inv = 1 - mask
    loss = F.l1_loss(output * mask_inv, target * mask_inv)
    return loss

def validation(model,device,validation_dataloader,criterion):
    val_loss = 0
    iter = 0
    model.eval()
    with torch.no_grad():
        for x,y in validation_dataloader:
            
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)
            
            y = y.permute(0, 1, 4, 2, 3)
            #x = x.permute(0, 1, 4, 2, 3)
            #x = sparsing(y.clone(),min_keep_ratio=0.08,max_keep_ratio=0.8) # This process sparses the already sparsed inputs on some differnet level of sparsness...
            x = sparsing(y.clone(),min_keep_ratio=0.65,max_keep_ratio=0.7,dropout=0.7) # prev: min_keep_ratio=0.08,max_keep_ratio=0.2
         
            x = rearrange(x, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
            pred_y = model(x)
            
            #x = rearrange(x, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
            y = rearrange(y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
            #pred_y = rearrange(pred_y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
            loss = criterion(pred_y, y) + masked_l1_loss(pred_y,y,mask=x)
    
            
            val_loss += loss.item()
            iter +=1

    return val_loss/iter


def train(name,dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50,load_checkpoint="",val_chunks = 1):
    torch.set_float32_matmul_precision('highest')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Device is: {device}')

    model = ViTInpainting(img_size=512,patch_size=8)
 
    #opt = torch.optim.Adam(diffmodel.model.parameters(), lr = learning_rate)
    opt = torch.optim.AdamW(model.parameters(), lr = learning_rate,weight_decay=1e-4)
    
    criterion = nn.MSELoss(reduction="mean")

    scheduler_type_cosine = True
    if scheduler_type_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, 
        T_0=10,     # Number of epochs before the first restart        
        T_mult=2,   # Multiplier for restart period
        eta_min=1e-7  # Minimum LR
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                       factor=0.8, patience=5, 
                                                       verbose=False,min_lr=1e-5)

    torch.backends.cudnn.benchmark = True
    model.to(device)

    losses = []
    val_losses = []
    
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    #    opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #opt = torch.optim.AdamW(diffmodel.model.parameters(), lr = 1e-5,weight_decay=1e-4)
    
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        val_losses = checkpoint['val_loss']
        best_loss = val_losses[-1]
        print(f'Validation loss is: {best_loss}')
        #best_loss = 1e8
    else:
        start_epoch = 1
        best_loss = 1e8
    
    for epoch in range(start_epoch,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        model.train()
        stime = time()
        chunk_loss = 0
        temp_val_losses = 0
        for i in range(0, number_of_chunks):
            if i % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            train_dataloader, _ = load_dataloader(i)

            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(model,device,train_dataloader,criterion)
                continue

            loader_loss = 0
            total = 0
            for step, (x,y) in enumerate(train_dataloader): 
                # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
                
                bs = y.shape[0]
                x, y = x.to(device), y.to(device)
                
                y = y.permute(0, 1, 4, 2, 3)
                
                #x = x.permute(0, 1, 4, 2, 3)
                x = sparsing(y.clone(),min_keep_ratio=0.65,max_keep_ratio=0.7,dropout=0.7) # This process sparses the already sparsed inputs on some differnet level of sparsness...

                # Here do: x reshaping...
                
                
                x = rearrange(x, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
                pred_y = model(x)

                accum_steps = 1
                
                y = rearrange(y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
                
                #pred_y = rearrange(pred_y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
                loss = (criterion(pred_y, y) + masked_l1_loss(pred_y,y,mask=x)) / accum_steps
                
                #opt.zero_grad()
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    #print("update..................")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    #torch.nn.utils.clip_grad_value_(model.model.parameters(), clip_value=0.1)

                    opt.step()
                    opt.zero_grad()
                #torch.nn.utils.clip_grad_norm_(diffmodel.model.parameters(), 1.0)
                #opt.step()
                
                loader_loss += loss.item() * accum_steps # so it wont mess up my diagrams of losses...
                total += 1
            
            chunk_loss += loader_loss/total

            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loader_loss/total:.6f}")
        losses.append(chunk_loss/number_of_chunks)

        val_losses.append(temp_val_losses/val_chunks)
        if scheduler_type_cosine:
            scheduler.step()
        else:
            scheduler.step(temp_val_losses)
        temp_val_losses = 0

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Training loss => {losses[-1]:.6f} - Validation Loss: {val_losses[-1]:.6f}")
        print(f'learning_rate: {scheduler.get_last_lr()}')
        if epoch % 5 == 0:
            if epoch >= 15 and epoch <= 120:
                update_loss_png(losses[14:], val_losses[14:],epoch, name)
            if epoch > 120:
                update_loss_png(losses[-120:], val_losses[-120:],epoch, name)

        if val_losses[-1] < best_loss:
            
            best_loss = val_losses[-1]
            print(f'New best model, with valdiation loss: {best_loss}')
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'val_loss': val_losses
            }
            torch.save(checkpoint, "checkpoints/" + name + ".pth")
        
        torch.cuda.empty_cache()


def predict(model_path, chunk_idx=0, num_images=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Load model
    model = ViTInpainting(img_size=512, patch_size=8)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)

    # Load a validation chunk
    _, val_loader = load_dataloader(chunk_idx)
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)

    # Take only `num_images`
    x = x[:num_images]
    y = y[:num_images]

    # Preprocess
    y = y.permute(0, 1, 4, 2, 3)  # (B, V, C, H, W)
    x_sparse = sparsing(y.clone(), min_keep_ratio=0.65, max_keep_ratio=0.7, dropout=0.7)

    x_input = rearrange(x_sparse, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
    y_target = rearrange(y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)

    with torch.no_grad():
        y_pred = model(x_input)

    # Move to CPU for visualization
    x_input = x_input.cpu()
    y_target = y_target.cpu()
    y_pred = y_pred.cpu()

    # Show results
    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(TF.to_pil_image(x_input[i]))
        axs[0].set_title('Input (sparse)')
        axs[1].imshow(TF.to_pil_image(y_pred[i].clamp(0,1)))
        axs[1].set_title('Prediction')
        axs[2].imshow(TF.to_pil_image(y_target[i]))
        axs[2].set_title('Target')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def setup_and_train(name,checkpoint=""):
    batch=8
    dataset_size = 1000 #the wjole dataset size is: 1464, for now...
    number_of_chunks = 32 # (1230*13/16)/1230 --> 81%
    val_chunks=3
    manage_dataloaders(dataset_size,number_of_chunks,batch)
    train(name,dataset_size,number_of_chunks,batch_size=batch,epochs=250,learning_rate=0.001,load_checkpoint=checkpoint,val_chunks=val_chunks)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_UNet(nn.Module):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768 // 8, depth=6, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Encoder
        self.start_dim = 12
        self.e1 = encoder_block(3, self.start_dim)
        self.e2 = encoder_block(self.start_dim, self.start_dim * 2)
        self.e3 = encoder_block(self.start_dim * 2, self.start_dim * 4)
        self.e4 = encoder_block_no_pool(self.start_dim * 4, embed_dim)

        # Bottleneck Transformer (shared across views)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_bottleneck = nn.TransformerEncoder(transformer_layer, num_layers=depth)

        # Decoder (mirroring the encoder)
        self.d4 = decoder_block(embed_dim, self.start_dim * 4)
        self.d3 = decoder_block(self.start_dim * 4, self.start_dim * 2)
        self.d2 = decoder_block(self.start_dim * 2, self.start_dim)
        self.d1 = decoder_block_no_upsample(self.start_dim, 3)  # Final RGB output

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.register_buffer('pos_embed', get_sinusoidal_positional_encoding(num_patches * 10, embed_dim, device='cpu'))  # overprovisioning

        # Final Transformer after UNet output
        final_layer = nn.TransformerEncoderLayer(d_model=3, nhead=3)
        self.final_transformer = nn.TransformerEncoder(final_layer, num_layers=2)

    def forward(self, x):
        """
        x: (B, V, 3, H, W) — V views of RGB images
        Returns:
            out: (B, V, 3, H, W)
        """
        B, V, C, H, W = x.shape
        feats = []

        # -----------------------------
        # 1. Encoder per view
        # -----------------------------
        skips = []
        for v in range(V):
            xv = x[:, v]  # (B, 3, H, W)
            f1, p1 = self.e1(xv)
            f2, p2 = self.e2(p1)
            f3, p3 = self.e3(p2)
            f4, p4 = self.e4(p3)  # (B, embed_dim, H', W')
            feats.append(p4)
            skips.append((f1, f2, f3))

        # (B, V, C, H', W') → (B, C, V * H' * W')
        cnn_feat = torch.stack(feats, dim=1)
        B, V, C, Hf, Wf = cnn_feat.shape
        tokens = cnn_feat.permute(0, 2, 1, 3, 4).reshape(B, C, -1).permute(0, 2, 1)

        # Positional encoding
        pos_embed = self.pos_embed[:, :tokens.size(1)].to(tokens.device)
        tokens = tokens + pos_embed

        # Transformer bottleneck
        tokens = tokens.permute(1, 0, 2)
        tokens = self.transformer_bottleneck(tokens)
        tokens = tokens.permute(1, 0, 2)

        # Back to (B, V, C, H', W')
        tokens = tokens.permute(0, 2, 1).reshape(B, C, V, Hf, Wf).permute(0, 2, 1, 3, 4)

        # -----------------------------
        # 2. Decoder per view
        # -----------------------------
        outs = []
        for v in range(V):
            t = tokens[:, v]  # (B, C, Hf, Wf)
            f1, f2, f3 = skips[v]

            d4 = self.d4(t, f3)
            d3 = self.d3(d4, f2)
            d2 = self.d2(d3, f1)
            d1 = self.d1(d2)  # (B, 3, H, W)

            outs.append(d1)

        out = torch.stack(outs, dim=1)  # (B, V, 3, H, W)

        # -----------------------------
        # 3. Final transformer on outputs
        # -----------------------------
        out_tokens = out.view(B, V, 3, -1).permute(0, 3, 1, 2).reshape(B * out.shape[-1] * out.shape[-2], V, 3)
        final = self.final_transformer(out_tokens)  # (tokens, B', 3)
        final = final.mean(dim=1)  # average over views
        final = final.view(B, 3, H, W)

        return final



















# random thingys....

def validation(model,device,validation_dataloader,criterion_mse,criterion_ssim):
    val_loss = 0
    iter = 0
    model.eval()
    with torch.no_grad():
        for x,y in validation_dataloader:
            
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)
            
            y = y.permute(0, 1, 4, 2, 3)
            x = x.permute(0, 1, 4, 2, 3)
                
            pred_y = model(x)
            
            y = rearrange(y, 'b views c h w -> b (c views) h w')
            
            #pred_y = rearrange(pred_y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)

            loss_mse = criterion_mse(pred_y, y)
            #loss_ssim = criterion_ssim(pred_y, y)
            loss = loss_mse# + loss_ssim
            val_loss += loss.item()
            iter +=1

    return val_loss/iter


import os
def prepare_sample(name,device,idx = 66):
    incomplete = np.load(os.path.join('wsl_prep_data', "incomplete_projections", f'{idx}.npy'))
    placeholder = incomplete[2,:,:,:].copy()
    incomplete[2,:,:,:] = incomplete[3,:,:,:]
    incomplete[3,:,:,:] = placeholder
    incomplete[2:4,:,:,:] = np.flip(incomplete[2:4,:,:,:],axis=2)
    incomplete = torch.tensor(incomplete, dtype=torch.float32).unsqueeze(0).to(device)

    torchshow.save(rearrange(incomplete[0], '(b1 b2) h w c -> c (b1 h) (b2 w)', b1=2, b2=2), f"./GAN_predictions/{name}_target.jpeg")
    return rearrange(incomplete, 'b views h w c -> b views c h w')


import gc
def train(name,dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50,load_checkpoint="",val_chunks = 1):
    torch.set_float32_matmul_precision('medium')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')

    model = Model()
 
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion_mse = nn.L1Loss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                       factor=0.1, patience=5, 
                                                       verbose=False,min_lr=1e-4)
    torch.backends.cudnn.benchmark = True
    model.to(device)

    losses = []            
    val_losses = []
    best_loss = 1e8
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        val_losses = checkpoint['val_loss']
    else:
        start_epoch = 1

    for epoch in range(start_epoch,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        model.train()
        stime = time()
        chunk_loss = 0
        temp_val_losses = 0
        for i in range(0, number_of_chunks):
            train_dataloader, _ = load_dataloader(i)

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(model,device,train_dataloader,criterion_mse)
                continue

            loader_loss = 0
            total = 0

            for it_, (x, y) in enumerate(train_dataloader):

                pred_y = model(x)
                opt.zero_grad()
                loss = criterion_mse(y, pred_y)
                
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) shoud I?.....
                opt.step()
                
                loader_loss += loss.item()

                total += 1
            

            chunk_loss += loader_loss/total
            
            #chunk_d_loss += d_loss_track/total
            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loader_loss/total:.4f}")
        
        losses.append(chunk_loss/number_of_chunks)
        val_losses.append(temp_val_losses/val_chunks)

        scheduler.step(temp_val_losses)
        temp_val_losses = 0

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Training loss => {losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")
        print(f'learning_rate: {scheduler.get_last_lr()}')

        if epoch % 5 == 0 or epoch == 12:
            update_loss_png(losses, val_losses,epoch, name)
            
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
        
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'val_loss': val_losses
            }
            torch.save(checkpoint, "checkpoints/" + name + ".pth")
    
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'losses': losses,
    'val_loss': val_losses
    }
    torch.save(checkpoint, "checkpoints/" + name + "_final.pth")


def update_loss_png(losses, val_loss,epoch, name):

    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Training Loss', alpha=0.4)    
    plt.plot(val_loss, label='Validation Loss', alpha=0.4)


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss and Validation Loss')

    # Save the plot
    plt.savefig(f"losses_pngs/{name}.png") # Something else comes here..............................
    plt.close()

def setup_and_train(name,checkpoint=""):
    batch=8
    dataset_size = 5000 #the wjole dataset size is: 1464, for now...
    number_of_chunks = 32 # (1230*13/16)/1230 --> 81%
    val_chunks=6
    manage_dataloaders(dataset_size,number_of_chunks,batch)
    train(name,dataset_size,number_of_chunks,batch_size=batch,epochs=5_000,learning_rate=2e-4,load_checkpoint=checkpoint,val_chunks=val_chunks)






if __name__ == '__main__':
    setup_and_train("0718_ViT",checkpoint="checkpoints/0718_ViT.pth") #checkpoints/0718_ViT.pth

    '''
    0718_ViT: trained till 51 epochs was around 0.026 the best....
    
    '''