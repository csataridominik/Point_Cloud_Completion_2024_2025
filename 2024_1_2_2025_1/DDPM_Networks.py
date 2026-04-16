import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# U-Net Architecture for Diffusion Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoding layers
        for feature in features:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature, feature, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ))
            in_channels = feature

        # Decoding layers
        for feature in reversed(features):
            self.decoder.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature, feature, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ))
        
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        for i, decode in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = decode(x)
        
        return self.final_layer(x)

# Diffusion Model Wrapper
class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.unet = unet
        
        self.beta_schedule = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
    def forward_diffusion(self, x0, t):
        """Forward diffusion: add noise to sparse projections"""
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def reverse_diffusion(self, x_noisy, t):
        """Reverse diffusion: Denoise sparse projections step-by-step"""
        predicted_noise = self.unet(x_noisy)
        beta_t = self.beta_schedule[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        
        mean = (1 / torch.sqrt(self.alpha[t])) * (x_noisy - beta_t * predicted_noise / torch.sqrt(1 - alpha_cumprod_t))
        noise = torch.randn_like(x_noisy) if t > 0 else torch.zeros_like(x_noisy)
        
        return mean + torch.sqrt(beta_t) * noise
    
    def forward(self, x):
        """Denoising using the learned model"""
        for t in reversed(range(self.timesteps)):
            x = self.reverse_diffusion(x, t)
        return x

# Training Loop
def train_diffusion(model, dataloader, optimizer, device, epochs=10):
    model.to(device)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for sparse_inputs, dense_targets in dataloader:
            sparse_inputs, dense_targets = sparse_inputs.to(device), dense_targets.to(device)
            
            t = torch.randint(0, model.timesteps, (sparse_inputs.shape[0],), device=device)
            x_noisy, noise = model.forward_diffusion(sparse_inputs, t)
            predicted_noise = model.unet(x_noisy)
            
            loss = mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
