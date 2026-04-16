#This code was created based on this article: https://medium.com/@adityanutakki/sr3-explained-and-implemented-in-pytorch-from-scratch-b43b9742c232

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_DDPM import manage_dataloaders,load_dataloader
from math import log
from time import time
import torchshow
from Original_IPT_Model import ImageDenoiser


class ImageDenoiser_POC(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_channels=3, embedding_dim=512, num_heads=8, num_layers=6):
        super(ImageDenoiser_POC, self).__init__()

        self.patch_dim = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Patch embedding
        self.linear_encoding = nn.Linear(patch_size * patch_size * num_channels, embedding_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Decoder
        self.linear_decoding = nn.Linear(embedding_dim, patch_size * patch_size * num_channels)

    def forward(self, x):
        # Convert image into patches
        x = F.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2)

        # Pass through embedding layer
        x = self.linear_encoding(x)

        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Convert back to image format
        x = self.linear_decoding(x)
        x = x.transpose(1, 2).contiguous()
        x = F.fold(x, output_size=128, kernel_size=self.patch_dim, stride=self.patch_dim)

        return x

def train(dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50):
    name = "20250327_IPT_Deeper"
    sample_input = np.load("XYZ_projections_sparse\XYZ_66.npy")
    sample_input = sample_input.transpose(0, 3, 1, 2).copy()
    sample_input = sample_input[0,:,:,:]
    #sample_input = (sample_input * 2) - 1  # Scale to [-1, 1]
    sample_input = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0)
    torchshow.save(sample_input, f"./torchshow_IPT/sample_input.jpeg")
    
    sample_target = np.load("XYZ_projections\XYZ_66.npy")
    sample_target = sample_target.transpose(0, 3, 1, 2).copy()
    sample_target = sample_target[0,:,:,:]
    #sample_target = (sample_target * 2) - 1  # Scale to [-1, 1]
    sample_target = torch.tensor(sample_target, dtype=torch.float32).unsqueeze(0)
    torchshow.save(sample_target, f"./torchshow_IPT/sample_target.jpeg")

    torch.set_float32_matmul_precision('medium')
    
    chunk_size = dataset_size // number_of_chunks
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')
    sample_input = sample_input.to(device) # sending the sample to device...
    
    model = ImageDenoiser()
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    torch.backends.cudnn.benchmark = True
    model.to(device)

    losses = []

    for epoch in range(1,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        model.train()
        stime = time()

        for i in range(0, number_of_chunks):
            train_dataloader, _ = load_dataloader(i)
            for x,y in train_dataloader: 
                # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
                
                bs = y.shape[0]
                x, y = x.to(device), y.to(device)
                
                y = y[:, 0, :, :, :]
                x = x[:, 0, :, :, :]  
                y = y.permute(0, 3, 1, 2)
                x = x.permute(0, 3, 1, 2)

                predicted_x = model(x)
                loss = criterion(y, predicted_x)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                losses.append(loss.item())

            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loss:.4f}")

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
        if epoch % 10 == 0 or epoch == 5 or epoch == epochs:
            sample(model,sample_input,device,epoch)

    torch.save(model.state_dict(), "trained_model/"+name+".pth")

    losses = np.asarray(losses).reshape(len(np.asarray(losses))//epoch, epoch)
    train_losses_avg = losses.mean(axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_avg, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    
    # Save the plot
    plt.savefig("losses_pngs/"+name+".png")
    plt.close()


def sample(model, lr_img, device,epoch):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    stime = time()
    with torch.no_grad():
        predicted_x = model(lr_img)
        
            
    ftime = time()
    torchshow.save(predicted_x, f"./torchshow_IPT/sample__at_epoch_{str(epoch)}.jpeg")
    print(f"Done denoising in {ftime - stime}s ")

batch=8
dataset_size = 1024
number_of_chunks = 32
manage_dataloaders(dataset_size,number_of_chunks,batch)
train(dataset_size,number_of_chunks,batch_size=batch,epochs=200,learning_rate=1e-4)
