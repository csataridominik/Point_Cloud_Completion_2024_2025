import torch
import torchshow

import numpy as np
from tqdm import tqdm
from FlowModel import FlowModel, flow_matching_loss
import scipy.integrate
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt



def load_images(size=128):
    transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
    ])

    '''
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="data", download=True, transform=transform),
        batch_size=32,
        shuffle=True
    )'''

    full_dataset = datasets.CIFAR10(root="data", download=True, transform=transform)
    ship_class_index = full_dataset.class_to_idx["cat"]
    print("Ship class index:", ship_class_index)  # should print 8

    # 4. Get the indices of all "ship" samples
    ship_indices = [i for i, (_, label) in enumerate(full_dataset) if label == ship_class_index]

    # 5. Create a subset dataset containing only ships
    ship_dataset = torch.utils.data.Subset(full_dataset, ship_indices)

    # 6. Create a DataLoader for just the ships
    ship_loader = torch.utils.data.DataLoader(ship_dataset, batch_size=32, shuffle=True)

    return ship_loader

def sample_source(batch):
    # Sample from a 2D standard Gaussian (mean=0, std=1)
    return torch.randn_like(batch)

def sample_target(batch_size):
    # Sample the x-coordinate uniformly in the range [-2, 2)
    x1 = torch.rand(batch_size) * 4 - 2

    # Sample the y-coordinate:
    # Step 1: draw from uniform [0, 1)
    # Step 2: subtract either 0 or 2, randomly (via torch.randint)
    # Result: values centered roughly around -2 or -1
    x2_ = torch.rand(batch_size) - torch.randint(high=2, size=(batch_size, )) * 2

    # Add a vertical shift depending on whether the x1 bin is even or odd
    # This creates the alternating row offset of the checkerboard
    x2 = x2_ + (torch.floor(x1) % 2)

    # Stack x1 and x2 into (batch_size, 2) vectors, and scale the whole grid
    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return torch.tensor(data, dtype=torch.float32)


import matplotlib.pyplot as plt

def show_states(gaussian_samples,checkerboard_samples):
    plt.figure(figsize=(12, 6))

    # Plot Gaussian source
    plt.subplot(1, 2, 1)
    plt.scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], alpha=0.5)
    plt.title("Gaussian Source Samples")
    plt.axis("equal")
    plt.grid(True)

    # Plot Checkerboard target
    plt.subplot(1, 2, 2)
    plt.scatter(checkerboard_samples[:, 0], checkerboard_samples[:, 1], alpha=0.5)
    plt.title("Checkerboard Target Samples")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train(name,dataloader, epochs=400, lr=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = []
    best_loss = 1e8

    model = FlowModel(in_channels=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            
            x0 = sample_source(imgs)
    
            # x1 = imgs
            t = torch.rand(imgs.size(0), 1).to(device)

            loss = flow_matching_loss(model, x0, imgs, t)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }

            torch.save(checkpoint, ".FllowMatching/" + name + ".pth")

@torch.no_grad()
def sample_flow(model, x0, n_steps=50, device="cuda"):
    model.eval()
    x = x0.clone().to(device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.ones(x.size(0), 1, device=device) * (i / n_steps)
        v1 = model(x, t)
        x_pred = x + v1 * dt
        t_next = torch.ones(x.size(0), 1, device=device) * ((i + 1) / n_steps)
        v2 = model(x_pred, t_next)
        x = x + 0.5 * (v1 + v2) * dt  # average slope
    return x

@torch.no_grad()
def sample_flow(model, x0, n_steps=50, device="cuda"):
    """
    Integrate learned flow using 4th-order Runge–Kutta (RK4).
    Args:
        model: trained FlowModel
        x0: starting image tensor (B, C, H, W)
        n_steps: number of integration steps
        device: "cuda" or "cpu"
    Returns:
        x: final integrated tensor (B, C, H, W)
    """
    model.eval()
    x = x0.clone().to(device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((x.size(0), 1), i / n_steps, device=device)
        k1 = model(x, t)

        t2 = torch.full_like(t, i / n_steps + 0.5 * dt)
        k2 = model(x + 0.5 * dt * k1, t2)

        k3 = model(x + 0.5 * dt * k2, t2)
        t3 = torch.full_like(t, i / n_steps + dt)
        k4 = model(x + dt * k3, t3)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

@torch.no_grad()
def sample_final(name, device="cuda", n_samples=16, img_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowModel(in_channels=3).to(device)
    checkpoint = torch.load(name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    """
    Generate final images by integrating from Gaussian noise (source) to data (target).
    """
    # Sample Gaussian source noise
    x0 = torch.randn(n_samples, 3, img_size, img_size, device=device)

    # Integrate through flow
    x1_hat = sample_flow(model, x0, n_steps=5_00, device=device)

    # Clamp/normalize output for visualization
    #x1_hat = (x1_hat.clamp(-1, 1) + 1) / 2  # scale to [0,1]

    # Make grid of generated samples
    grid = vutils.make_grid(x1_hat, nrow=4)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.title("Generated Images (Flow Matching)")
    plt.show()

if __name__ == "__main__":
    #show_states(sample_source(1000),sample_target(1000))
    dataloader = load_images()
    
    predict = False
    if predict:
        sample_final(".FllowMatching/flowmatching_ships.pth")
    else:
        train("flowmatching_ships",dataloader)
