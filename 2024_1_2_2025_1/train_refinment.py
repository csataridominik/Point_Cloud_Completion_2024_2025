import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchshow
from time import time
import numpy as np
from tqdm import tqdm
from MultiViewUNet import DiffusionModel

from MultiViewDiffusionModel import sparsing,update_loss_png,sample_ddim
import open3d as o3d
from dataset import manage_dataloaders,load_dataloader
import torchshow as ts

def load_model(load_checkpoint,device,time_steps = 1000,learning_rate=1e-4):
    ddpm = DiffusionModel(time_steps = time_steps)
    opt = torch.optim.Adam(ddpm.model.parameters(), lr = learning_rate)
    
    checkpoint = torch.load(load_checkpoint, map_location=device)
    ddpm.model.load_state_dict(checkpoint['model_state_dict'])
    curr_epoch = checkpoint['epoch'] + 1
    print(f'Trained for {curr_epoch} epochs...')
    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return ddpm

def reproject(all_images,epsilon = 0.2):
    coordinates = []
    
    count = 0
    # All images is shape [B,C,H,W] we iterate through H and W
    for i in range(all_images.shape[-1]):
        for j in range(all_images.shape[-2]):
            x, y, z = all_images[:,i, j]
            
            if abs(x) > epsilon and abs(y) > epsilon and abs(z) > epsilon:
                count += 1
                coordinates.append([x, y, z])
    
    coordinates = np.array(coordinates)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(coordinates)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2)

    filtered_pcd = pcd.select_by_index(ind,invert=False)

    o3d.visualization.draw_geometries([filtered_pcd])
    
    '''
    outlier_cloud = pcd.select_by_index(ind,invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])
    '''
    return np.asarray(filtered_pcd.points)

def reproject2(coords1, coords2):
    # Create the first point cloud and assign blue color
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(coords1)
    pcd1.paint_uniform_color([0, 0, 1])  # Blue

    # Create the second point cloud and assign red color
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(coords2)
    pcd2.paint_uniform_color([1, 0, 0])  # Red

    # Visualize both point clouds together
    o3d.visualization.draw_geometries([pcd1, pcd2])

def DDIM_predict_and_save(model,device):
    load_path = "XYZ_projections"
    save_path = "DDIM_predictions"
    sidx=0
    fidx=1024
    for idx in tqdm(range(sidx,fidx)):
        sample_input = np.load(f"{load_path}\XYZ_{str(idx)}.npy")
        sample_input = sample_input.transpose(0, 3, 1, 2).copy()
        sample_input = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0)
        sample_input = sparsing(sample_input,max_keep_ratio=0.6) # Perhapse should put sparsing in train? or is ti okay here...
        sample_input = (sample_input * 2) - 1  # Scale to [-1, 1]
        sample_input[:,2:4,:,:,:] = torch.flip(sample_input[:,2:4,:,:,:],dims=[4])
        pred = sample_ddim(model=model,device=device,lr_img=sample_input)

        pred = pred.cpu().detach().numpy()
        np.save(save_path+f"\\{str(idx)}.npy", pred)

def train_refinment(name,device,number_of_chunks,epochs=100,learning_rate=1e-4):
    
    torch.set_float32_matmul_precision('medium')

    model = DnCNN()
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

                bs = y.shape[0]
                x, y = x.to(device), y.to(device)
                x = x.squeeze()
                pred_y = model(x)
                loss = criterion(pred_y, y)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                losses.append(loss.item())

            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loss:.4f}")

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
        
        if epoch % 5 == 0:
            update_loss_png(losses,epoch,name)
      
    torch.save(model.state_dict(), "trained_model/"+name+".pth")

def main_refinment(name,model_path = "checkpoints/2025_0512_epoch_3500.pt",create_predicitons=False): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if create_predicitons:
        model = load_model(model_path,device)
        DDIM_predict_and_save(model,device)

    setup_and_train(name,device)

def setup_and_train(name,device):
    batch=8
    dataset_size = 1024 #the wjole dataset size is: 1464, for now...
    number_of_chunks = 16
    manage_dataloaders(dataset_size,number_of_chunks,batch)
    train_refinment(name,device,number_of_chunks,epochs=50,learning_rate=1e-4)

def normalize(x):
    for i in range(x.shape[0]):
        x_min = x[i].min()
        x_max = x[i].max()
        x[i] = (x[i] - x_min) / (x_max - x_min + 1e-8)
    return x

import matplotlib.pyplot as plt

def visualize_xyz(x_vals,y_vals,z_vals):
    """
    all_images: torch.Tensor or np.ndarray of shape [3, H, W]
    Visualizes histograms of x, y, z channels (entire data, no filtering).
    """


    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(x_vals, bins=500, range=(-1, 3), color='red', alpha=0.7)
    plt.title('X Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(y_vals, bins=500, range=(-1, 3), color='green', alpha=0.7)
    plt.title('Y Value Distribution')
    plt.xlabel('Value')

    plt.subplot(1, 3, 3)
    plt.hist(z_vals, bins=500, range=(-1, 3), color='blue', alpha=0.7)
    plt.title('Z Value Distribution')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.show()

import numpy as np

def peak_based_filter(data, bin_width=0.01, distance_thresh=0.15):
    # Histogram
    bins = np.arange(min(data), max(data) + bin_width, bin_width)
    hist, bin_edges = np.histogram(data, bins=bins)

    # Find bin with maximum frequency
    peak_index = np.argmax(hist)
    peak_center = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2

    # Filter based on distance from peak
    return np.abs(data - peak_center) < distance_thresh

def filter_out_distant_coords(all_images):
    x_vals = all_images[0].flatten()
    y_vals = all_images[1].flatten()
    z_vals = all_images[2].flatten()
    
    mask_0 = (x_vals != 0) & (y_vals != 0) & (z_vals != 0)
    #x_vals = x_vals[mask_0]
    #y_vals = y_vals[mask_0]
    #z_vals = z_vals[mask_0]
    
    mask_x = peak_based_filter(x_vals, bin_width=0.0001, distance_thresh=0.15)
    mask_y = peak_based_filter(y_vals, bin_width=0.0001, distance_thresh=0.15)
    mask_z = peak_based_filter(z_vals, bin_width=0.0001, distance_thresh=0.15)

    combined_mask = mask_x | mask_y | mask_z

    filtered_xyz = np.stack([x_vals[~combined_mask],
                            y_vals[~combined_mask],
                            z_vals[~combined_mask]], axis=1)
    

    # Filter outliers before normalization...
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2)

    filtered_pcd = pcd.select_by_index(ind,invert=False)
    return np.asarray(filtered_pcd.points)


def final_prediction(model,device,path,ratio):

    sample_input = np.load(f"wsl_new_data_test\input\{path}.npy")
    placeholder = sample_input[2,:,:,:].copy()
    sample_input[2,:,:,:] = sample_input[3,:,:,:]
    sample_input[3,:,:,:] = placeholder
    #sample_input[1,:,:,:] = np.flip(sample_input[1,:,:,:])
    sample_input[2:4,:,:,:] = np.flip(sample_input[2:4,:,:,:],axis=2)
    
    sample_target = np.load(rf"wsl_new_data_test\target\target_{path}.npy")
    placeholder = sample_target[2,:,:,:].copy()
    sample_target[2,:,:,:] = sample_target[3,:,:,:]
    sample_target[3,:,:,:] = placeholder
    #sample_input[1,:,:,:] = np.flip(sample_input[1,:,:,:])
    sample_target[2:4,:,:,:] = np.flip(sample_target[2:4,:,:,:],axis=2)
    sample_target = (sample_target * 2) - 1
    sample_target = sample_target.transpose(0, 3, 1, 2).copy()
    sample_target = torch.tensor(sample_target, dtype=torch.float32).unsqueeze(0)

    sample_input = sample_input.transpose(0, 3, 1, 2).copy()
    sample_input = (sample_input * 2) - 1  # Scale to [-1, 1]
    sample_input = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0)
    
    pred = sample_ddim(model=model,device=device,lr_img=sample_input,ddim_steps=100)

    #sample_input = normalize(sample_input[0])  
    sample_input = rearrange(sample_input[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
    torchshow.show(sample_input)
    torchshow.show(pred)

    sample_target = rearrange(sample_target[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=2)
    torchshow.show(sample_target)
    return pred,sample_target.cpu().detach().numpy()

import pickle
def save_pcd_list(X,y,ratio):
    with open('wsl_cd_loss_transfer/X_'+str(ratio)[-1]+'.pkl', 'wb') as f:
        pickle.dump(X, f)
    
    with open('wsl_cd_loss_transfer/y_'+str(ratio)[-1]+'.pkl', 'wb') as f:
        pickle.dump(y, f)

def predict_and_reproject(model_path="checkpoints/2025_0512_epoch_3500.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_path,device)

    pcd_sizes = [1,2,3]
    sidx = 0#1237
    fidx = 400

    pcd_original = 0
    pcd_pred = 0
    for n in pcd_sizes:
        print(f'Calculating for n: {n}...')
        X = []
        y = []
        for i in tqdm(range(sidx,fidx)):

            pred,target = final_prediction(model,device,str(i),n)
            filtered_pred = filter_out_distant_coords(pred[0].cpu().detach().numpy())
            
            pcd_pred = (filtered_pred - filtered_pred.min()) / (filtered_pred.max() - filtered_pred.min())
            
            '''
            
            Rather than taking the min max: take the min of the last 5 min or max or more outlier detection......
            I can also do: after reprojection statistical otulier to normaize again.. that would solve....
            
            '''
            
            # This part is for Histogram visualisation:
            #visualize_xyz(pcd_pred[:,0],pcd_pred[:,1],pcd_pred[:,2])#pred[0].cpu().detach().numpy()
            
            #pcd_pred = reproject(pred_norm) # (pred[0].cpu().detach().numpy()+1)/2-0.5
            pcd_original = reproject(target)
            min_val = pcd_pred.min()
            max_val = pcd_pred.max()
            
            #print(f'This is min and max: {min_val}, and {max_val}')  

            #reproject2(pcd_pred,pcd_original)
            X.append(pcd_pred)
            y.append(pcd_original)
            
        save_pcd_list(X,y,n)
        sidx += 400
        fidx += 400


from MVPCC_Networks import InpaintGenerator as GAN
def predict_and_reproject_GAN(model_path="checkpoints/2025_0512_epoch_3500.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GAN()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    pcd_sizes = [1,2,3]
    sidx = 0#1237
    fidx = 400

    pcd_original = 0
    pcd_pred = 0
    for n in pcd_sizes:
        print(f'Calculating for n: {n}...')
        X = []
        y = []
        for i in tqdm(range(sidx,fidx)):

            sample_input = np.load(f"wsl_new_data_test\input\{i}.npy")
            placeholder = sample_input[2,:,:,:].copy()
            sample_input[2,:,:,:] = sample_input[3,:,:,:]
            sample_input[3,:,:,:] = placeholder
            sample_input[2:4,:,:,:] = np.flip(sample_input[2:4,:,:,:],axis=2)
            
            target = np.load(rf"wsl_new_data_test\target\target_{i}.npy")
            placeholder = target[2,:,:,:].copy()
            target[2,:,:,:] = target[3,:,:,:]
            target[3,:,:,:] = placeholder
            target[2:4,:,:,:] = np.flip(target[2:4,:,:,:],axis=2)
            target = (target * 2) - 1
            target = target.transpose(0, 3, 1, 2).copy()
            #target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

            sample_input = sample_input.transpose(0, 3, 1, 2).copy()
            sample_input = (sample_input * 2) - 1  # Scale to [-1, 1]
            sample_input = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0)

            model.eval()
            pred = model(sample_input) 

            pred = rearrange(pred[0], '(c views) h w -> views c h w', views=4, c=3)
            pred_show = rearrange(pred, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1 = 2, b2 = 2)
            target_show = rearrange(target, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1 = 2, b2 = 2)
            
            torchshow.show(rearrange(sample_input[0], '(b1 b2) c h w -> c (b1 h) (b2 w)', b1 = 2, b2 = 2))
            torchshow.show(target_show)
            torchshow.show(pred_show)
            # from here already written code part...
            
            '''
            filtered_pred = filter_out_distant_coords(pred[0].cpu().detach().numpy())
            
            pcd_pred = (filtered_pred - filtered_pred.min()) / (filtered_pred.max() - filtered_pred.min())
            '''
            reproject(pred_show.cpu().detach().numpy())

            X.append(pcd_pred)
            y.append(pcd_original)
            

        save_pcd_list(X,y,n)
        sidx += 400
        fidx += 400

# -------------------------------- This part is only for predicting real world data -------------------------------------
def load_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Lines defining vertices
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def create_pcd(points):
    # Get vertex coordinates (Nx3 array)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    normalized_points = (points - min_vals) / (max_vals - min_vals)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(normalized_points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2)

    filtered_pcd = pcd.select_by_index(ind,invert=False)

    o3d.visualization.draw_geometries([filtered_pcd])


import matplotlib.pyplot as plt
def simple_visualize_no_background(all_images):
    points = all_images.reshape(-1, 3)  # Flatten all points
    
    # Filter out points that are exactly (1,1,1)
    mask = ~np.all(points == 0, axis=1)  # keep points NOT equal to (1,1,1)
    filtered_points = points[mask]
    print(f'Thesse are the number of points: { filtered_points.shape}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    o3d.visualization.draw_geometries([pcd])

def reproject(coords):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd])

import os
if __name__ == "__main__":
    
    #target = torch.load("o3d_draw/target_0.pt").squeeze(0).detach().cpu().numpy()
    #incomplete = torch.load("o3d_draw/0.pt").squeeze(0).detach().cpu().numpy()
    
    reproject(np.load("o3d_draw/target.npy"))
    reproject(np.load("o3d_draw/in.npy"))
    reproject(np.load("o3d_draw/out.npy"))
    #reproject(np.load("o3d_draw/out_coarse.npy"))
    
    #predict_and_reproject(model_path="checkpoints/2025_0603.pth")
    #all_images = np.load("real_data/test_ship.npy")
    #simple_visualize_no_background(all_images)

   # predict_and_reproject_GAN(model_path="checkpoints/20250929_MVPCC_model_InPainting.pth")
    '''
        prev check: checkpoints/2025_0613_250ts_lowsparsity.pth

    '''
    #points = load_obj('real_data/reallife.obj')
    #print(points.shape)

    #create_pcd(obj)
    #main_refinment(name="refinment_20250519",create_predicitons=False)
    
    #reproject(final_pred)
