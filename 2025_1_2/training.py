# %% [markdown]
# Imports

# %%
from pytorch3d.loss import chamfer_distance
import torch
from torch import nn
import numpy as np
import gc
#from models import PoinTr
from time import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    look_at_view_transform,
    SoftPhongShader,
    PointLights,
    AmbientLights,
    SoftGouraudShader
)

from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset,DataLoader
import pickle

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import torch   
from pytorch3d.ops import sample_points_from_meshes
import os


#from geomloss import SamplesLoss
#sinkhorn = SamplesLoss("sinkhorn", p=1, blur=0.05)

#from PointAttn_BigBird import Model
#from PointAttN import Model
#from BigBird_v2_model import Model
#from pcn_encoder import Model
#from density_aware_pcn import Model

from RNN01 import Model

class Args:
    def __init__(self):
        self.dataset = 'pcn'
        self.base_dim = 32
        self.refine_dim = 64
        self.refine_ratio1 = 2
        self.refine_ratio2 = 4
        self.refine_channels = 64


# %% [markdown]
# 0. Dataloader Management

# %%

class IncompleteDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        # Return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single data point and its label
        data_point = self.X[idx]
        label = self.y[idx]
        return data_point, label

# Currently no validation or test set is used, only using data for training...
def create_dataloader(dataset,batch_size = 32, train_ratio=0.9):
 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,0

def load_dataloader(processed_idx):
    file_path = 'dataloaders/train' + str(processed_idx) + '.pkl'

    with open(file_path, 'rb') as file:
        dataloader_train = pickle.load(file)

    return dataloader_train

def save_dataloader(dataloader,processed_idx,type):
    with open('dataloaders/'+type+str(processed_idx)+'.pkl', 'wb') as f:
        pickle.dump(dataloader, f)
    

def manage_dataloaders(dataset_size,number_of_chunks = 60,batch_size = 4):
    chunk_size = dataset_size // number_of_chunks

    for i in tqdm(range(number_of_chunks)):
        dataset = create_dataset(i*chunk_size,(i+1)*chunk_size)

        train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)
        save_dataloader(train_dataloader,i,"train")
        #save_dataloader(test_dataloader,i,"test")

def create_dataset(it,dataset_size):

    X = []
    y = []

    for i in range(it,dataset_size):
        GT_XYZ = torch.load(os.path.join("target", f'target_{i}.pt'))
        incomplete = torch.load(os.path.join("incomplete_projections", f'{i}.pt'))#.unsqueeze(0)

        #GT_XYZ,_ = sample_farthest_points(GT_XYZ, K=1024,random_start_point=False)
        #incomplete,_ = sample_farthest_points(incomplete, K=5120,random_start_point=False)

        y.append(GT_XYZ) 
        X.append(incomplete)

    X = torch.stack(X).float()
    y = torch.stack(y).float()

    dataset = IncompleteDataset(X, y)
    return dataset


# %%
def sparsing(dense_tensor, min_keep_ratio=0.3, max_keep_ratio=1, dropout = 1):
    """
    Keeps a random subset of non-zero (object) pixels per [batch, view],
    applying the same mask across all 3 channels, with others set to 0.

    Inputs:
        dense_tensor: [B, V, C, H, W] tensor with RGB data
    Returns:
        sparsified_tensor: same shape, sparsely masked
    """
    
    V, C, H, W = dense_tensor.shape
    sparsified_tensor = torch.full_like(dense_tensor, -1)
    
    #mask_size = np.random.randint(0,2)
    
    views = np.random.choice(4, size=dropout, replace=False)
    #else:
    #    views = np.random.choice(4, size=3, replace=False)

    #print(f'This is views: {views}')

    #view = np.random.randint(0,4)
    #view = -1
    for v in range(V):
            
        #if v == view and np.random.rand() <= dropout:                
        if v in views:
            sparsified_tensor[v] = 0 # broadcasts... masks out one of the inputs
            continue
        
        img = dense_tensor[v]  # [C, H, W]
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
        
        sparsified_tensor[v] = img * mask
    
    return sparsified_tensor

def render_mesh_vertex_colors_normalized(idx,viz=False,shift=45,elevation=15):
    obj_path = "/home/fv1tw4/pytorch3d_test_20250521/meshes/" + str(idx) + ".obj"
    image_size = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load mesh
    verts, faces, _ = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # GLOBAL NORMALIZATION
    '''min_val = verts.min()
    max_val = verts.max()
    verts_normalized = (verts - min_val) / (max_val - min_val + 1e-8)'''

    # Create vertex-color texture (RGB = globally normalized XYZ)
    textures = TexturesVertex(verts_features=verts[None])
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

    # Configure rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # Compute bounding sphere center and radius
    center = 0.5 * (verts.max(0).values + verts.min(0).values)
    radius = (verts - center).norm(dim=1).max()

    # Set field of view and compute tight camera distance
    fov = 30.0  # degrees; decrease to zoom in more
    dist = radius.item() / np.sin(np.radians(fov / 2))
    dist *= 1.05  # Add slight margin to avoid clipping

    #shift = 45 # THIS IS CHAGNED SO ITS GIVEN BY ARG.

    # Generate 4 viewpoints
    Rs, Ts = look_at_view_transform(
        dist=dist,
        elev=elevation,
        azim=[0+shift, 90+shift, 180+shift, 270+shift],
        at=center[None].cpu()
    )

    # Ambient-only lighting
    lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))

    all_images = []
    for i in range(4):
        cameras = FoVPerspectiveCameras(
            device=device,
            R=Rs[i][None],
            T=Ts[i][None],
            fov=fov
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftGouraudShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        # Render
        images = renderer(mesh)
        image_rgba = images[0].cpu().numpy()
        rgb = image_rgba[..., :3]
        alpha = image_rgba[..., 3]

        rgb[alpha == 0] = 0
        #rgb = np.clip(rgb, 0, 1)
        
        all_images.append(rgb)
        if viz:
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb)
            plt.title(f"Azimuth: {90*i}°")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"projection_{i}.png")
            plt.show()

    # Save all projections
    all_images = np.array(all_images)
    #np.save(f"projections/{idx}.npy", all_images)
    return all_images

def render_pointcloud_vertex_colors_normalized(verts,idx, viz=True):

    #obj_path = f"/home/fv1tw4/pytorch3d_test_20250521/meshes/{idx}.obj"
    image_size = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load .obj file (only use verts, ignore faces)
    #verts, _, _ = load_obj(obj_path, load_textures=False)
    verts = verts.to(device)

    # GLOBAL NORMALIZATION
    '''
    min_val = verts.min()
    max_val = verts.max()
    verts_normalized = (verts - min_val) / (max_val - min_val + 1e-8)
   '''
    # Use normalized XYZ as RGB
    colors = verts  # (N, 3)
    pointcloud = Pointclouds(points=[verts], features=[colors])

    # Bounding sphere for camera distance
    center = 0.5 * (verts.max(0).values + verts.min(0).values)
    radius = (verts - center).norm(dim=1).max()
    fov = 30.0
    dist = radius.item() / np.sin(np.radians(fov / 2)) * 1.05
    shift = 45

    Rs, Ts = look_at_view_transform(
        dist=dist,
        elev=15,
        azim=[0 + shift, 90 + shift, 180 + shift, 270 + shift],
        at=center[None].cpu(),
    )

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,  # Tune this
        points_per_pixel=10,
    )

    all_images = []
    for i in range(4):
        cameras = FoVPerspectiveCameras(
            device=device, R=Rs[i][None], T=Ts[i][None], fov=fov
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(),
        )

        images = renderer(pointcloud)
        rgb = images[0, ..., :3].cpu().numpy()

        #rgb = np.clip(rgb, 0, 1)
        all_images.append(rgb)

        if viz:
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb)
            plt.title(f"Azimuth: {90*i}°")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"projection_{idx}_{i}.png")
            plt.show()

    all_images = np.array(all_images)
    #os.makedirs("incomplete_projections", exist_ok=True)

def load_npy(idx):
    GT_XYZ = np.load("projections/" +str(idx)+".npy")
    return GT_XYZ

def create_pcd(index,device):
    
    shift = random.randint(0, 360)
    elevation = random.randint(0, 360)
    pcd = render_mesh_vertex_colors_normalized(index,viz=False,shift=shift,elevation=elevation)
    pcd = torch.from_numpy(pcd).to(device)
    pcd = pcd.permute(0, 3, 1, 2)        
    return pcd

def load_pcd_and_fps(device,idx,target_size):
    
    mesh = load_objs_as_meshes([f"meshes/{idx}.obj"], device=device)
    points = sample_points_from_meshes(mesh, target_size)
    return points

def save_pcd(pcd,idx):
    torch.save(pcd,f"incomplete_projections/{idx}.pt")

def proj_2_arr(proj,incomplete_shape_size=4096):
    coords = proj.permute(0, 2, 3, 1).reshape(-1, 3)

    # Filter out (0, 0, 0)
    non_bg = coords[~(coords == 0).all(dim=1)]

    # Get unique coordinates
    unique_coords = torch.unique(non_bg, dim=0)
        
    unique_coords = unique_coords.cuda().unsqueeze(0)
    fps_points, _ = sample_farthest_points(unique_coords, K=incomplete_shape_size)
    fps_points = fps_points.squeeze(0)      # [4096, 3]

    return fps_points

def genrate_dataset():
        
    training = 1700
    from_idx = 0
    to_idx = 1700
    # repeating this:
    repeat = 10
    unique_array = list(range(training*repeat))
    random.shuffle(unique_array)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = True
    test = False

    size_of_predictions = 8192
    if train:
        for idx in tqdm(range(from_idx, to_idx)):
            target_pcd = (load_pcd_and_fps(device,idx,target_size = size_of_predictions)+1)/2
            
            
            for _ in range(repeat):
                
                pcd = (create_pcd(idx,device)+1)/2
                pcd_sparse1 = sparsing(pcd,dropout=1,min_keep_ratio=1,max_keep_ratio=1)
                '''
                pcd = (create_pcd(idx,device)+1)/2
                pcd_sparse2 = sparsing(pcd,dropout=1,min_keep_ratio=1,max_keep_ratio=1)
                
                pcd = (create_pcd(idx,device)+1)/2
                pcd_sparse3 = sparsing(pcd,dropout=2,min_keep_ratio=1,max_keep_ratio=1)
                
                pcd = (create_pcd(idx,device)+1)/2
                pcd_sparse_extra = sparsing(pcd,dropout=2,min_keep_ratio=1,max_keep_ratio=1)
                '''
                id1 = unique_array.pop()
                #id2 = unique_array.pop()
                #id3 = unique_array.pop()
                #id4 = unique_array.pop()

                # Needs to be saved....
                save_pcd(proj_2_arr(pcd_sparse1),id1)
                #save_pcd(proj_2_arr(pcd_sparse2), id2)
                #save_pcd(proj_2_arr(pcd_sparse3), id3)
                #save_pcd(proj_2_arr(pcd_sparse_extra),id4)
                
                torch.save(target_pcd, f"target/target_{id1}.pt")
                #torch.save(target_pcd, f"target/target_{id2}.pt")
                #torch.save(target_pcd, f"target/target_{id3}.pt")
                #torch.save(target_pcd, f"target/target_{id4}.pt")



#from Transformer import knn_point
def generate_dataset_knn():
            
    training = 1700
    from_idx = 0
    to_idx = 1700
    # repeating this:
    repeat = 5
    unique_array = list(range(training*repeat))
    random.shuffle(unique_array)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = True
    test = False

    size_histogram = []
    size_of_predictions = 16384

    if train:
        for idx in tqdm(range(from_idx, to_idx)):
            target_pcd = load_pcd_and_fps(device,idx,target_size = size_of_predictions)
            
            for _ in range(repeat):            
                pcd = target_pcd.clone()
                
                rand_idx = torch.randint(0, size_of_predictions, (1,)).item()  # [B, S=1]
                new_xyz = pcd[:, rand_idx, :].unsqueeze(0)  # [B, 1, 3]

                # 2️⃣ Get neighbors (KNN)
                group_idx = knn_point(nsample=size_of_predictions//4, xyz=pcd, new_xyz=new_xyz)  # [B, 1, 50]
                group_idx = group_idx.view(-1)  # flatten indices

                # 3️⃣ Remove those points to make a new sparse PCD
                mask = torch.ones(size_of_predictions, dtype=torch.bool)
                mask[group_idx] = False
                sparse_pcd = pcd[:, mask, :]  # your new sparse point cloud
                
                id1 = unique_array.pop()

                torch.save(sparse_pcd.squeeze(0), f"incomplete_projections/{id1}.pt")
                torch.save(target_pcd.squeeze(0), f"target/target_{id1}.pt")

                #torch.save(target_pcd, f"target/target_{id2}.pt")
                #torch.save(target_pcd, f"target/target_{id3}.pt")
                #torch.save(target_pcd, f"target/target_{id4}.pt")
  

def cd_loss(pc1,pc2,norm_=2):
    loss,_ = chamfer_distance(pc1,pc2,norm=norm_)

    return loss.mean()


def emd(pc1,pc2): #L2
    pc1 = pc1.contiguous()
    pc2 = pc2.contiguous()
    
    loss = sinkhorn(pc1,pc2)
    return loss.mean()


def combined_loss(pred,target,pred_coarse,y_coarse):
    J_0 = cd_loss(pred,target,norm_=1)
    J_1 = cd_loss(pred_coarse,target,norm_=1)
    #J_1 = emd(pred_coarse,y_coarse)
    return J_0 + J_1

def validation(model,device,validation_dataloader):
    val_loss = 0
    iter = 0
    model.eval()
    with torch.no_grad():
        for x,y in validation_dataloader:
            
            x,_ = sample_farthest_points(x, K=4096,random_start_point=False)
            #y = (y + 1) / 2 #normalize it to [0,1]
            #x = (x + 1) / 2 #normalize it to [0,1]
            x, y = x.to(device).transpose(2,1), y.to(device)
            #y_coarse,_ = sample_farthest_points(y, K=512,random_start_point=False)
            loss = model(x,y,is_training=True) # jsut because we need the same loss
   
            #loss = combined_loss(pred_y, y,pred_coarse,y_coarse)
            val_loss += loss.item()
            iter +=1

    return val_loss/iter

import torch

def load_pretrained_and_freeze(freeze=False):
    #ckpt_path = "checkpoints/model_pcn.pth"
    ckpt_path = "checkpoints/PointAttN_BigBird_v2.pth"
    
    ckpt = torch.load(ckpt_path)
    pretrained_dict = ckpt['model_state_dict']#ckpt["net_state_dict"]

    # --- 1️⃣ Initialize your model ---
        
    class Args:
        def __init__(self):
            self.dataset = 'pcn'  # or 'c3d'
    args = Args()
    model = Model(args).cuda()
    model_dict = model.state_dict()

    # --- 2️⃣ Filter pretrained weights that match shape ---
    matched_weights = {}
    skipped_weights = []
    for k, v in pretrained_dict.items():
        key = k.replace("module.", "")
        if key in model_dict:
            if v.shape == model_dict[key].shape:
                matched_weights[key] = v
            else:
                skipped_weights.append(key)
        else:
            skipped_weights.append(key)

    print(f"✅ Loaded {len(matched_weights)} compatible parameters.")
    if skipped_weights:
        print(f"⚠️ Skipped {len(skipped_weights)} mismatched parameters:\n", skipped_weights)

    # --- 3️⃣ Load matched weights ---
    model_dict.update(matched_weights)
    model.load_state_dict(model_dict, strict=False)

    # --- 4️⃣ Optionally freeze attention layers ---
    if freeze:
        frozen_count = 0
        for name, param in model.named_parameters():
            if "attn" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"🔒 Frozen {frozen_count} attention parameters.")

    print("✅ Model ready (pretrained weights loaded where possible).")
    return model

def unfreeze_attention(model,lr):
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
            print(f"🔓 Unfroze {name}")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    return opt

def train(name,dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50,load_checkpoint="",val_chunks = 1):
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')

    model = Model()

 

    #model = load_pretrained_and_freeze()

    opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                       factor=0.8, patience=10, 
                                                       verbose=False,min_lr=1e-4)
    '''
    
    #scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=9.5e-6, max_lr=8e-5, step_size_up=100)

    torch.backends.cudnn.benchmark = True
    model.to(device)

    losses = []            
    val_losses = []
    best_loss = 1e8
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        #opt.load_state_dict(checkpoint['optimizer_state_dict'])
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
            train_dataloader = load_dataloader(i)
            '''
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            '''

            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(model,device,train_dataloader)
                continue

            loader_loss = 0
            total = 0

            for it_, (x, y) in enumerate(train_dataloader):
                
                x,_ = sample_farthest_points(x, K=4096,random_start_point=False)
                #y = (y + 1) / 2 #normalize it to [0,1]
                #x = (x + 1) / 2 #normalize it to [0,1]
                #print(f'This is minmax of x: {x.min()} - {x.max()}')
                #print(f'This is minmax of y: {y.min()} - {y.max()}')
                x, y = x.to(device).transpose(2,1), y.to(device)
                
                #coarse_pred,pred_y = model(x)
                opt.zero_grad()
                
                loss = model(x,y,is_training=True)
                

                #y_coarse,_ = sample_farthest_points(y, K=512,random_start_point=False)
                #loss = combined_loss(y, pred_y,coarse_pred,y_coarse)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #shoud I?.....??
                opt.step()
                
                loader_loss += loss.item()
                

                total += 1
                
                #scheduler.step() # THIS IS FOR CYVLEIC!!!!!!!!!!!!!!!!!!!
            chunk_loss += loader_loss/total
            
            #chunk_d_loss += d_loss_track/total
            #print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {loader_loss/total:.4f}")
        
        losses.append(chunk_loss/number_of_chunks)
        val_losses.append(temp_val_losses/val_chunks)

        #scheduler.step(temp_val_losses)
        temp_val_losses = 0

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Training loss => {losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")
        #print(f'learning_rate: {scheduler.get_last_lr()}')

        if epoch % 5 == 0 or epoch == 1:
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

def  setup_and_train(name,config,setup=True, checkpoint="",):
    batch = 16
    dataset_size = 1700*5
    number_of_chunks = 16 
    val_chunks=2
    if setup:
        manage_dataloaders(dataset_size,number_of_chunks,batch)
    train(name,dataset_size,number_of_chunks,batch_size=batch,epochs=1_000,learning_rate=1e-4,load_checkpoint=checkpoint,val_chunks=val_chunks)

class Config:
    trans_dim = 64*2    # Transformer embedding dimension
    knn_layer = 4        # KNN layers in PCTransformer
    num_pred = 1024*4     # number of fine points to generate
    num_query = 256     # number of query (coarse) points
    num_heads = 4
    depth = [4, 4]

if __name__ == "__main__":

    config = Config()
    setup_and_train("model_december_2",config,setup=True,checkpoint="")
    # PointAttN_BigBird_v10_mlp_25m this has no selfatttention in encoder...
    # PointAttN_BigBird_v5 This was actually v2 type....... but improved at 50 epochs for some reason

    # baseline is baseline: PointAttN_BigBird_baseline 
    # Epoch [306/5000] trained in 214.25990056991577s; Training loss => 0.0497 - Validation Loss: 0.1282
    # step1 step2 = 2,2, 2048 base

    # PointAttN_BigBird_test_runs current with two heads
    # PointAttN_BigBird_v4 here I have replaced global procedure with ffn 
    # PointAttN_BigBird_v3 : 8 heads, 4 global, 0 local, 8 random
    # PointAttN_BigBird_v2 is in progress, xformers were used... 20251110 checkpoints/PointAttN_BigBird_v2.pth
    #  is around L1 combined loss: 0.0435, 4 heads, 2 local, 6 random, 2 global tokens
    # PointAttN_v1 os working!
    # PointAttN_v1_low_lr is also working, bit better? at elast lower loss...
    #  PointAttN_v1 is going to be the overwritten to have 16384 as target and 4096 as input
    # generate_dataset_knn()
