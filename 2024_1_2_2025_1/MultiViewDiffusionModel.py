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
#from MultiViewUNet import DiffusionModel
from DiT_0904 import DiffusionModel
import gc

from torch.utils.data import Dataset,DataLoader
import os

class LazyNPYDataset(Dataset):
    def __init__(self, indices, projection_dir, incomplete_dir):
        self.indices = indices
        self.projection_dir = projection_dir
        self.incomplete_dir = incomplete_dir

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        GT_XYZ = np.load(os.path.join(self.projection_dir, f'target_{real_idx}.npy'))
        incomplete = np.load(os.path.join(self.incomplete_dir, f'{real_idx}.npy'))

        # process swap/flip if needed
        placeholder = GT_XYZ[2,:,:,:].copy()
        GT_XYZ[2,:,:,:] = GT_XYZ[3,:,:,:]
        GT_XYZ[3,:,:,:] = placeholder
        GT_XYZ[2:4,:,:,:] = np.flip(GT_XYZ[2:4,:,:,:],axis=2)

        # same for incomplete
        placeholder = incomplete[2,:,:,:].copy()
        incomplete[2,:,:,:] = incomplete[3,:,:,:]
        incomplete[3,:,:,:] = placeholder
        incomplete[2:4,:,:,:] = np.flip(incomplete[2:4,:,:,:],axis=2)

        return torch.from_numpy(incomplete).float(), torch.from_numpy(GT_XYZ).float()



def validation(diffmodel,device,validation_dataloader,criterion):
    val_loss = 0
    iter = 0
    diffmodel.model.eval()
    with torch.no_grad():
        for x,y in validation_dataloader:
            
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)
            
            y = y.permute(0, 1, 4, 2, 3)
            x = x.permute(0, 1, 4, 2, 3)
            #x = sparsing(y.clone(),min_keep_ratio=0.08,max_keep_ratio=0.8) # This process sparses the already sparsed inputs on some differnet level of sparsness...
            #x = sparsing(y.clone(),min_keep_ratio=0.65,max_keep_ratio=0.7,dropout=0.7) # prev: min_keep_ratio=0.08,max_keep_ratio=0.2

            x = (x * 2) - 1   # Converts [0,1] → [-1,1]
            y = (y * 2) - 1   
            

            ts = torch.randint(low = 1, high = diffmodel.time_steps, size = (bs, ))

            gamma = diffmodel.alpha_hats[ts].to(device)
            ts = ts.to(device = device)

            y, target_noise = diffmodel.add_noise(y, ts)

            # ----------------------
            target_noise = target_noise.permute(0,1,4,2,3)
            target_noise = target_noise.reshape(target_noise.shape[0],12,256,256)
            z = torch.cat([x, y], dim=-1)   # shape: [4, 4, 256, 256, 6]

            # 2. Move channels to second position
            z = z.permute(0, 1, 4, 2, 3)    # shape: [4, 4, 6, 256, 256]

            # 3. Merge the "4" and "6" into 24
            z = z.reshape(z.shape[0], 24, 256, 256)  # final shape: [4, 24, 256, 256]

            
            predicted_noise = diffmodel.model(z, gamma)
            loss = criterion(target_noise, predicted_noise)
            
            val_loss += loss.item()
            iter +=1

    return val_loss/iter


def train(name,dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50,time_steps=2000,load_checkpoint="",val_chunks = 1):
    torch.set_float32_matmul_precision('highest')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Device is: {device}')

    diffmodel = DiffusionModel(time_steps = time_steps)
 
    #opt = torch.optim.Adam(diffmodel.model.parameters(), lr = learning_rate)
    opt = torch.optim.AdamW(diffmodel.model.parameters(), lr = learning_rate,weight_decay=1e-4)
    
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
                                                       factor=0.9, patience=10, 
                                                       verbose=False,min_lr=1e-6)

    torch.backends.cudnn.benchmark = True
    diffmodel.model.to(device)

    losses = []
    val_losses = []
    
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        diffmodel.model.load_state_dict(checkpoint['model_state_dict'])
    #    opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #opt = torch.optim.AdamW(diffmodel.model.parameters(), lr = 1e-5,weight_decay=1e-4)
    
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        val_losses = checkpoint['val_loss']
        best_loss = val_losses[-1]
        #best_loss = 1e8
        print(f'Validation loss is: {best_loss}')
        
    else:
        start_epoch = 1
        best_loss = 1e8
    
    for epoch in range(start_epoch,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        diffmodel.model.train()
        stime = time()
        chunk_loss = 0
        temp_val_losses = 0
        for i in range(0, number_of_chunks):
            if i % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            #train_dataloader, _ = load_dataloader(i)
#-----------------------------------------------
# later should change top streaming dataset....
            start = i * dataset_size//number_of_chunks
            end = min((i + 1) * dataset_size//number_of_chunks, dataset_size)
            indices = list(range(start, end))

            dataset = LazyNPYDataset(indices, 'wsl_prep_data/target', 'wsl_prep_data/incomplete_projections')
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#-----------------------------------------------



            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(diffmodel,device,train_dataloader,criterion)
                continue

            loader_loss = 0
            total = 0
            for step, (x,y) in enumerate(train_dataloader): 
                # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
                
                bs = y.shape[0]
                x, y = x.to(device), y.to(device)

                y = y.permute(0, 1, 4, 2, 3)
                x = x.permute(0, 1, 4, 2, 3)
                #x = sparsing(y.clone(),min_keep_ratio=0.65,max_keep_ratio=0.7,dropout=0.7) # This process sparses the already sparsed inputs on some differnet level of sparsness...

                x = (x * 2) - 1   # Converts [0,1] → [-1,1]
                y = (y * 2) - 1   
                
                # flipping the input so it has shapes at the right aprt of the image..
                #y[:, 2:4, :, :, :] = torch.flip(y[:, 2:4, :, :, :], dims=[4])
                #x[:, 2:4, :, :, :] = torch.flip(x[:, 2:4, :, :, :], dims=[4])

                ts = torch.randint(low = 1, high = diffmodel.time_steps, size = (bs, ))

                gamma = diffmodel.alpha_hats[ts].to(device)
                ts = ts.to(device = device)

                y, target_noise = diffmodel.add_noise(y, ts)
                target_noise = target_noise.permute(0,1,4,2,3)
                target_noise = target_noise.reshape(target_noise.shape[0],12,256,256)
                z = torch.cat([x, y], dim=-1)   # shape: [4, 4, 256, 256, 6]

                # 2. Move channels to second position
                z = z.permute(0, 1, 4, 2, 3)    # shape: [4, 4, 6, 256, 256]

                # 3. Merge the "4" and "6" into 24
                z = z.reshape(z.shape[0], 24, 256, 256)  # final shape: [4, 24, 256, 256]

                #predicted_noise = diffmodel.model(x,y, gamma)
                predicted_noise = diffmodel.model(z, gamma)
                
                accum_steps = 1  
                loss = criterion(target_noise, predicted_noise) / accum_steps
                
                #opt.zero_grad()
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    #print("update..................")
                    #torch.nn.utils.clip_grad_norm_(diffmodel.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_value_(diffmodel.model.parameters(), clip_value=1)

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
            'model_state_dict': diffmodel.model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'val_loss': val_losses
            }
            torch.save(checkpoint, "checkpoints/" + name + ".pth")
        
        torch.cuda.empty_cache()


def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def update_loss_png(losses, val_loss,epoch, name):

    plt.figure(figsize=(12, 6))
    if epoch > 120:
        plt.plot(range(epoch-120,epoch),losses, label='Training Loss', alpha=0.4)
        plt.plot(range(epoch-120,epoch),val_loss, label='Validation Loss', alpha=0.4)
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

def sample(model, lr_img, device,epoch,sparse = False):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    stime = time()
    with torch.no_grad():
    
        y = torch.randn_like(lr_img, device = device)
        lr_img = lr_img.to(device)
        for i, t in enumerate(range(model.time_steps - 1, 0 , -1)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
    
            t = torch.tensor(t, device = device).long()
            pred_noise = model(lr_img, y, alpha_t_hat.view(-1).to(device),pass_ViT=False)
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
    
    ftime = time()
    y = rearrange(y[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
    y = (y + 1) / 2 # scaling back to [0,1] for vis porpuses....
    if sparse:
        torchshow.save(y, f"./torchshow_sample_0512/blank_1200/sample__at_epoch_{str(epoch)}.jpeg")

    else:
        torchshow.save(y, f"./torchshow_sample_0512/sample__at_epoch_{str(epoch)}.jpeg")
        torchshow.save(y, f"./torchshow_sample_0512/current.jpeg")

    print(f"Done denoising in {ftime - stime}s ")

def noising_tuning(time_steps=1001,stepsize=25):
    sample_target = np.load("XYZ_projections\XYZ_66.npy")
    sample_target = sample_target.transpose(0, 3, 1, 2).copy()
    sample_target = (sample_target * 2) - 1  # Scale to [-1, 1]
    
    sample_target[2:4] = np.flip(sample_target[2:4], axis=3)

    sample_target = torch.tensor(sample_target, dtype=torch.float32).unsqueeze(0)

    diffmodel = DiffusionModel(time_steps = time_steps)

    for idx in range(0, time_steps, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        sample, noise = diffmodel.add_noise(sample_target,t)
        
        sample = (sample.clamp(-1, 1) + 1) / 2
        sample = rearrange(sample[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
        torchshow.save(sample, f"./tuning_noise/ts_{idx}.jpeg")

def setup_and_train(name,checkpoint=""):
    batch=8
    dataset_size = 6000 # used to be 6000 #the wjole dataset size is: 1464, for now...
    number_of_chunks = 32 # (1230*13/16)/1230 --> 81%
    val_chunks=8
    #manage_dataloaders(dataset_size,number_of_chunks,batch)
    train(name,dataset_size,number_of_chunks,batch_size=batch,time_steps=250,epochs=20_000,learning_rate=0.0007,load_checkpoint=checkpoint,val_chunks=val_chunks)
 
def predict(model_name,time_steps,name,idx=""):
    
    sample_target = np.load(f"XYZ_projections\XYZ_{str(idx)}.npy")
    sample_target = sample_target.transpose(0, 3, 1, 2).copy()
    sample_target = rearrange(sample_target, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
    torchshow.save(sample_target, f"./diffusion_predicitons/{name}_target.jpeg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffmodel = DiffusionModel(time_steps = time_steps)
    
    diffmodel.model.load_state_dict(torch.load(f"trained_model/{model_name}.pth"))
    diffmodel.model.to(device)
    diffmodel.model.eval()

    if idx == "":
        obj = torch.zeros([3,128,128])
    else:
        obj = np.load(f"XYZ_projections_sparse\XYZ_{str(idx)}.npy")
        obj = obj.transpose(0, 3, 1, 2).copy()
        obj = (obj * 2) - 1  # Scale to [-1, 1]
 
    obj = torch.tensor(obj, dtype=torch.float32).unsqueeze(0)
    obj.to(device)
    obj[:, 2:4, :, :, :] = torch.flip(obj[:, 2:4, :, :, :], dims=[4])

    with torch.no_grad():
    
        y = torch.randn_like(obj, device = device)
        obj = obj.to(device)
        for i, t in enumerate(range(diffmodel.time_steps -1, 0 , -1)):
            alpha_t, alpha_t_hat, beta_t = diffmodel.alphas[t], diffmodel.alpha_hats[t], diffmodel.betas[t]
    
            t = torch.tensor(t, device = device).long()
            pred_noise = diffmodel.model(torch.cat([obj, y], dim = 2), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
            
    #y = (y.clamp(-1, 1) + 1) / 2 # Change this!!!!
    y = rearrange(y[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
    torchshow.save(y, f"./diffusion_predicitons/{name}.jpeg")

def sparsing(dense_tensor, min_keep_ratio=0.08, max_keep_ratio=0.9, dropout = 0.0):
    """
    Keeps a random subset of non-zero (object) pixels per [batch, view],
    applying the same mask across all 3 channels, with others set to 0.

    Inputs:
        dense_tensor: [B, V, C, H, W] tensor with RGB data
    Returns:
        sparsified_tensor: same shape, sparsely masked
    """
    B, V, C, H, W = dense_tensor.shape
    sparsified_tensor = torch.full_like(dense_tensor, -1)
    
    mask_size = np.random.randint(0,2)
    #mask_size = dropout

    if mask_size == 0:
        views = np.random.choice(4, size=1, replace=False)
    elif mask_size == 1:
        views = np.random.choice(4, size=2, replace=False)
    #else:
    #    views = np.random.choice(4, size=3, replace=False)

    #print(f'This is views: {views}')
    for b in range(B):
        #view = np.random.randint(0,4)
        #view = -1
        for v in range(V):
               
            #if v == view and np.random.rand() <= dropout:                
            if v in views:
                sparsified_tensor[b,v] = 0 # broadcasts... masks out one of the inputs
                continue
            
            img = dense_tensor[b, v]  # [C, H, W]
            # Object mask: any channel non-zero
            object_mask = (img > 0).any(dim=0)  # [H, W]
            num_obj = object_mask.sum().item()
            if num_obj == 0:
                continue
            
            keep_ratio = torch.rand(1).item() * (max_keep_ratio - min_keep_ratio) + min_keep_ratio
            num_keep = int(keep_ratio * num_obj)
            
            # Generate random binary mask for object pixels
            flat_mask = torch.zeros(num_obj, device=img.device)
            if num_keep > 0:
                flat_mask[:num_keep] = 1
                flat_mask = flat_mask[torch.randperm(num_obj)]
            
            # Create and apply the mask
            mask = torch.zeros(H, W, device=img.device)
            mask[object_mask] = flat_mask
            mask = mask.unsqueeze(0).expand(C, -1, -1)  # Expand to [C, H, W]
            
            sparsified_tensor[b, v] = img * mask
    
    
    return sparsified_tensor

def make_ddim_schedule(total_steps, ddim_steps):
    return np.round(np.linspace(0, total_steps - 1, ddim_steps)).astype(int)

@torch.no_grad()
def sample_ddim_infer(model, device, lr_img, ddim_steps=50):    
    return sample_ddim(model, device, lr_img, ddim_steps)


@torch.no_grad()
def sample_ddim(model,device, lr_img, ddim_steps=50):    
    model.to(device)
    model.eval()

    #stime = time()
    total_steps = model.time_steps
    ddim_timesteps = make_ddim_schedule(total_steps, ddim_steps)

    y = torch.randn_like(lr_img, device=device)
    lr_img = lr_img.to(device)

    for i in reversed(range(len(ddim_timesteps))):
        t = ddim_timesteps[i]
        t_tensor = torch.tensor(t, device=device).long()

        alpha_t = model.alphas[t]
        alpha_hat_t = model.alpha_hats[t]
        alpha_prev = model.alphas[ddim_timesteps[i - 1]] if i > 0 else model.alphas[0]

        # Predict noise
        pred_noise = model(lr_img, y, alpha_hat_t.view(-1).to(device))

        # Estimate x0
        x0_pred = (y - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_t)

        # DDIM update rule (eta = 0, deterministic)
        y = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise

    #ftime = time()
    
    y = rearrange(y, 'b (b1 b2) c h w -> b c (b1 h) (b2 w)', b1=2, b2=2)

    return y

from piqa import SSIM
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


if __name__ == "__main__":
    #noising_tuning()
    setup_and_train(name="0904_asd",checkpoint="checkpoints/0904_asd.pth") #checkpoints/2025_0613_250ts.pth
    
    '''
    checkpoints/0813.pth
    # checkpoints/0724_new_model.pth This was the new trasformer multi pathway model learned till 3600 epochs having 0.0022 minimum...

    # checkpoints/0719_new_model.pth: This one was on the nem model, but it stuck aorund 0.006 which is not great....

    checkpoints/2025_0613_250ts.pth --> This mdoel worked with sinusoidal embedding now i have changed it to nn.embedding(),
      where ts is directly used

    This is what I have trained prev:
    
    checkpoints/2025_0613.pth --> used 32 for norm, and have used 1000 timsteps. I have read, less is okay, I am trying 250

    '''

    #predict(model_name="2025_0424",idx=1300,time_steps=1000,name="test")

