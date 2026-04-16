#This code was created based on this article: https://medium.com/@adityanutakki/sr3-explained-and-implemented-in-pytorch-from-scratch-b43b9742c232

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset import manage_dataloaders,load_dataloader
from math import log
from time import time
import torchshow
from MVPCC_Networks import InpaintGenerator, Discriminator



# --------------------------------------------------------------------------------------------------------------
def total_variation_loss(x):
    """
    Total variation loss to encourage spatial smoothness.
    Assumes input x is of shape (batch_size, channels, height, width)
    """
    loss_tv = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
              torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    return loss_tv

@torch.no_grad
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

    model = InpaintGenerator()
 
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion_mse = nn.L1Loss(reduction="mean")
    criterion_ssim = SSIM(window_size=11,n_channels=12).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                       factor=0.1, patience=5, 
                                                       verbose=False,min_lr=1e-4)
    torch.backends.cudnn.benchmark = True
    model.to(device)

    sample = prepare_sample(name,device,idx = 66)
    # adv part:
    discriminator = Discriminator(in_channels=12).to(device)  # Adjust in_channels to match output
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    adversarial_loss = nn.BCELoss()

    losses = []            # total generator loss (mse + adv)
    val_losses = []
    g_adv_losses = []      # just adversarial loss
    d_losses = []          # discriminator loss
    best_loss = 1e8
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        val_losses = checkpoint['val_loss']
        g_adv_losses = checkpoint['g_adv_losses']
        d_losses = checkpoint['d_losses']
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
        start_epoch = 1
    prev_d_loss = 0
    for epoch in range(start_epoch,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        model.train()
        stime = time()
        chunk_loss = 0
        chunk_g_adv_loss = 0
        chunk_d_loss = 0
        temp_val_losses = 0
        for i in range(0, number_of_chunks):
            train_dataloader, _ = load_dataloader(i)

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(model,device,train_dataloader,criterion_mse,criterion_ssim)
                continue

            g_adv_loss_track = 0
            d_loss_track = 0
            loader_loss = 0
            total = 0
            #for x,y in train_dataloader: 
            disc_update = 0
            for it_, (x, y) in enumerate(train_dataloader):
                # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
                disc_update += 1
                bs = y.shape[0]
                x, y = x.to(device), y.to(device)
                
                y = y.permute(0, 1, 4, 2, 3)
                x = x.permute(0, 1, 4, 2, 3)

                pred_y = model(x)

                #y_real = rearrange(y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)
                y_real = rearrange(y, 'b views c h w -> b (c views) h w')
                
                y_fake = pred_y.clone()
                
                # Common losses here: ---------------------------

                # === Train Discriminator ===
                if it_ % 3 == 0:
                    discriminator.zero_grad()

                    # Real
                    real_out, _ = discriminator(y_real.detach())
                    real_labels = torch.ones_like(real_out)
                    loss_real = adversarial_loss(real_out, real_labels)

                    # Fake
                    fake_out, _ = discriminator(y_fake.detach())
                    fake_labels = torch.zeros_like(fake_out)
                    loss_fake = adversarial_loss(fake_out, fake_labels)

                    d_loss = loss_real + loss_fake
                    d_loss.backward()
                    opt_D.step()

                    d_loss_track += d_loss.item()
                # === Train Generator ===

                opt.zero_grad()
                fake_out, _ = discriminator(y_fake)
                target_labels = torch.ones_like(fake_out)
                g_adv_loss = adversarial_loss(fake_out, target_labels)

                loss_mse = criterion_mse(y_fake, y_real)
                
                #loss_tv = total_variation_loss(y_fake)
                #loss = 1 * loss_mse + 0.1 * g_adv_loss + 0.1/10 * loss_tv This was loss for first model in September
                # 20251003
                # mse helyett xyz coord diff
                #loss_ssim = criterion_ssim(y_fake, y_real)
                loss = 0.2 * loss_mse + 0.8 * g_adv_loss
                #loss = 0.8*loss_mse + 0.4 * g_adv_loss  # Add weight for GAN loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                loader_loss += loss.item()
                g_adv_loss_track += g_adv_loss.item()

                total += 1
            
            chunk_loss += loader_loss/total
            chunk_g_adv_loss += g_adv_loss_track/total

            chunk_d_loss += d_loss_track/(total / 3)

            #chunk_d_loss += d_loss_track/total
            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loader_loss/total:.4f} - g: {g_adv_loss_track/total:.4f} - d: {d_loss_track/total:.4f}")
        
        losses.append(chunk_loss/number_of_chunks)
        g_adv_losses.append(chunk_g_adv_loss / number_of_chunks)
        d_losses.append(chunk_d_loss / number_of_chunks)
        val_losses.append(temp_val_losses/val_chunks)

        scheduler.step(temp_val_losses)
        temp_val_losses = 0

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Training loss => {losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")
        print(f'learning_rate: {scheduler.get_last_lr()}')

        if epoch % 5 == 0 or epoch == 12:
            update_loss_png(losses, val_losses,epoch, name)
            update_loss_png(g_adv_losses, d_losses,epoch, "adv_d_and_g",True)
            
            predict(model,device,sample,epoch)
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
        
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'val_loss': val_losses,
            'g_adv_losses': g_adv_losses,
            'd_losses': d_losses,
            'discriminator_state_dict': discriminator.state_dict()
            }
            torch.save(checkpoint, "checkpoints/" + name + ".pth")
    
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'losses': losses,
    'val_loss': val_losses,
    'g_adv_losses': g_adv_losses,
    'd_losses': d_losses,
    'discriminator_state_dict': discriminator.state_dict()
    }
    torch.save(checkpoint, "checkpoints/" + name + "_final.pth")


def update_loss_png(losses, val_loss,epoch, name,adv=False):

    plt.figure(figsize=(12, 6))
    if adv:
        plt.plot(losses, label='g_adv_losses', alpha=0.4)    
        plt.plot(val_loss, label='d_losses', alpha=0.4)
    else:
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


from piqa import SSIM
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

def predict(model,device,sparse_pcd,epoch):
    model.eval()
    with torch.no_grad():
        pred_y = model(sparse_pcd)
        pred_y = rearrange(pred_y[0], '(c views) h w -> views c h w', views=4, c=3)
        
        pred_y = rearrange(pred_y, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1 = 2, b2 = 2)
        torchshow.save(pred_y, f"GAN_predictions/curr_image_at_{epoch}.jpeg")

if __name__ == "__main__":
    #noising_tuning()
    setup_and_train(name="20251001_MVPCC_model_InPainting_different_loss",checkpoint="") #checkpoints/2025_0606_GAN.pth
    # checkpoints/20250929_MVPCC_model_InPainting.pth

    #predict(model_name="2025_0424",idx=1300,time_steps=1000,name="test")

