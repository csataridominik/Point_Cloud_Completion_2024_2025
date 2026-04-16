import torch
import open3d as o3d
import os
#from models import PoinTr
from training import Config,combined_loss
import numpy as np
#from PointAttn_BigBird import Model
#from BigBird_v2_model import Model

from RNN01 import Model

from pytorch3d.ops import sample_farthest_points

class Args:
    def __init__(self):
        self.dataset = 'pcn'  # or 'c3d'

def reproject(coords):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd])


def predict(model_name,idx,save = True,pointr=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the model:
    if pointr:
        config = Config()
        model = PoinTr(config)
    else:

        model = Model()

    model.to(device)
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2) Load the Data:
    GT_XYZ = torch.load(os.path.join("target", f'target_{idx}.pt')).unsqueeze(0)
    incomplete = torch.load(os.path.join("incomplete_projections", f'{idx}.pt')).unsqueeze(0)
    incomplete,_ = sample_farthest_points(incomplete, K=4096,random_start_point=False)
    print(f'Size of incomplete input: {incomplete.shape}')
    print(f'Size of Ground Truth: {GT_XYZ.shape}')
    
    # 3) Predict:
    model.eval()
    with torch.no_grad():

        out,cd_loss = model(incomplete.transpose(2,1),GT_XYZ,is_training=False)
        """         print("\nEvaluation mode:")
        print(f"Coarse output shape: {out['out1'].shape}")
        print(f"Fine output shape: {out['out2'].shape}")
        print(f"Chamfer coarse (L2/L1): {out['cd_p_coarse'].item():.6f}, {out['cd_t_coarse'].item():.6f}")
        print(f"Chamfer fine (L2/L1): {out['cd_p'].item():.6f}, {out['cd_t'].item():.6f}")
        pred = out['out2']
        print(f'min max of target: {GT_XYZ.min()} and {GT_XYZ.max()}')
        print(f'min max of out: {pred.min()} and {pred.max()}') """

        print(f"Chamfer fine (L2): {cd_loss.item():.6f}")
        
        print(f"Fine output shape: {out.shape}")

        if save:
            np.save("o3d_draw/target.npy",GT_XYZ.squeeze(0).detach().cpu().numpy())
            np.save("o3d_draw/in.npy",incomplete.squeeze(0).detach().cpu().numpy())
            #np.save("o3d_draw/out.npy",out['out2'].squeeze(0).detach().cpu().numpy())
            #np.save("o3d_draw/out_coarse.npy",out['out1'].squeeze(0).detach().cpu().numpy())
            np.save("o3d_draw/out.npy",out.squeeze(0).detach().cpu().numpy())

        

if __name__ == "__main__":
    print('Open3D', o3d.__version__)
        
    predict("checkpoints/model_december_2.pth",605,save=True)