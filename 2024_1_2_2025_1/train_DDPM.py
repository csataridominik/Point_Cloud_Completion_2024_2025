import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import pickle
from dataset import create_dataset2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Channel reduction
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            
        self.att = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        self.conv = DoubleConv(out_channels * 2, out_channels)  # Handles concatenated features

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.channel_conv(self.up(x1))  # Reduce channels after upsampling
        else:
            x1 = self.up(x1)
            
        x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels=16, n_classes=12, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder with channel control
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, mask):
        # Input processing (unchanged)
        B, N, C, H, W = x.shape
        x_flat = x.view(B, N*C, H, W)
        mask_flat = mask.view(B, N, H, W)
        x_in = torch.cat([x_flat, mask_flat], dim=1)
        
        # Encoder
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with fixed channel flow
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits.view(B, N, C, H, W)    

# Modified Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=500):
        super().__init__()
        self.timesteps = timesteps
        self.unet = unet
        
        # Noise schedule remains same
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward_diffusion(self, x0, mask, t,sparse_inputs):
        # x0: [B, 4, 3, H, W], mask: [B, 4, 1, H, W]
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_bar[t].view(-1,1,1,1,1)
        noisy = alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise
        
        # Apply mask
        #noisy = noisy * (1 - mask) + x0 * mask
        noisy = noisy * (1 - mask) + sparse_inputs * mask
        return noisy, noise

# Modified Training Function
def train_diffusion(model, dataloader, optimizer, device):
    mse_loss = nn.MSELoss()
    
    model.train()
    total_loss = 0.0
        
    for sparse_inputs, dense_targets in dataloader:
        # Convert inputs to [B, 4, 3, H, W]
        sparse_inputs = sparse_inputs.view(-1, 4, 3, 256, 256).to(device)
        dense_targets = dense_targets.view(-1, 4, 3, 256, 256).to(device)
#        sparse_inputs = sparse_inputs.to(device)
#        dense_targets = dense_targets.to(device)

        # Create mask [B, 4, 1, H, W]
        mask = (sparse_inputs.abs().sum(dim=2, keepdim=True) > 0).float()
        
        t = torch.randint(0, model.timesteps, (sparse_inputs.shape[0],), device=device)
        x_noisy, noise = model.forward_diffusion(dense_targets, mask, t,sparse_inputs)
        predicted_noise = model.unet(x_noisy, mask)
        
#        loss = mse_loss(predicted_noise, noise)
        mask_expanded = mask.expand_as(predicted_noise)
        loss = mse_loss(predicted_noise * (1 - mask_expanded), noise * (1 - mask_expanded))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_image(XYZ,name):
    R = (255 * (XYZ / np.max(XYZ))).astype(np.uint8)
    R = np.transpose(R, (1, 2, 0))
    img = Image.fromarray(R, mode='RGB')  
    img.save(name)



def train(dataset_size,number_of_chunks = 60,learning_rate=0.001,epochs = 50,batch_size = 4,schedular=False):
    name = "20250315_DiffusionModel_Conditioning_01"

    torch.set_float32_matmul_precision('medium')
    
    chunk_size = dataset_size // number_of_chunks
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')
    unet = UNet().to(device)
    
    torch.backends.cudnn.benchmark = True
    model = DiffusionModel(unet).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=1)
    #optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    train_losses = []
    test_losses = []

    # For seeing progress.....
    test_output = np.load("XYZ_projections_sparse\XYZ_66.npy")
    test_output = test_output.transpose(0, 3, 1, 2).copy()


    for epoch in range(1,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        for i in range(0, number_of_chunks):

            train_dataloader, test_dataloader = load_dataloader(i)

            train_loss = train_diffusion(model,train_dataloader,optimizer,device)
            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {train_loss:.4f}")

            train_losses.append(train_loss)
                        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_losses[-1]:.4f}")
        
        '''
        if epoch % 10 == 0 or epoch == 1:
            unet.eval()

            views = torch.split(test_output, 3, dim=1)
            mask_per_view = [v.abs().sum(dim=1, keepdim=True) > 0 for v in views]
            mask = torch.cat(mask_per_view, dim=1).float()

            # Expand mask to match input channels (4 views -> 12 channels)
            mask = mask.repeat_interleave(3, dim=1)  # [B,12,H,W]
            with torch.no_grad():
                out = predict(unet,device,target=test_output,mask=mask)
            out = out.cpu().detach().numpy().squeeze()  # Convert to NumPy array
            out = out.reshape(4, 3, 256,256)[0]
            save_image(out,"test_images\\"+name+"\\"+str(epoch)+".png")
            unet.train()'''

        # Create one projection output:
         
    torch.save(model.state_dict(), "trained_model/"+name+".pth")

    train_losses = np.asarray(train_losses).reshape(number_of_chunks, epoch)
    #test_losses = np.asarray(test_losses).reshape(number_of_chunks, epoch)

    # Compute the mean along the first axis (averaging over the 32 chunks)
    train_losses_avg = train_losses.mean(axis=0)
    #test_losses_avg = test_losses.mean(axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_avg, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    
    # Save the plot
    plt.savefig("losses_pngs/"+name+".png")
    plt.close()

# From here already written parts:-------------------------------------------

def create_dataloader(dataset,batch_size = 32, train_ratio=0.9):
    
    train_size = int(train_ratio * len(dataset))
    '''
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    '''
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = 0
    return train_loader,test_loader

def load_dataloader(processed_idx):
    file_path = 'dataloaders/train' + str(processed_idx) + '.pkl'

    with open(file_path, 'rb') as file:
        dataloader_train = pickle.load(file)

    file_path = 'dataloaders/test' + str(processed_idx) + '.pkl'

    with open(file_path, 'rb') as file:
        dataloader_test = pickle.load(file)

    return dataloader_train, dataloader_test


def save_dataloader(dataloader,processed_idx,type):
    with open('dataloaders/'+type+str(processed_idx)+'.pkl', 'wb') as f:
        pickle.dump(dataloader, f)
    
def manage_dataloaders(dataset_size,number_of_chunks = 60,batch_size = 4):
    chunk_size = dataset_size // number_of_chunks
    

    for i in tqdm(range(number_of_chunks)):

        dataset = create_dataset2(i*chunk_size,(i+1)*chunk_size)
        
        train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)
        save_dataloader(train_dataloader,i,"train")
        save_dataloader(test_dataloader,i,"test")


batch=4
dataset_size = 256
number_of_chunks = 32
#manage_dataloaders(dataset_size,number_of_chunks,batch)
#train(dataset_size,number_of_chunks,batch_size=batch,epochs=500)



