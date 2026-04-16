import torch
import open3d as o3d
import torchshow as ts
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings


def pinhole_camera_projection(points):

    # Use the XYZ coordinates as features
    features = points.clone()

    # Create the point cloud
    point_cloud = Pointclouds(points=[points], features=[features])

    # Define camera positions (e.g., front, side, top, isometric)
    camera_positions = torch.tensor([
        [0.0, 0.0, -5.0],  # Front view
        [5.0, 0.0, 0.0],   # Side view
        [0.0, 5.0, 0.0],   # Top view
        [5.0, 5.0, -5.0],  # Isometric view
    ], dtype=torch.float32)

    # All cameras look at the origin
    R, T = look_at_view_transform(eye=camera_positions, at=((0, 0, 0),), up=((0, 1, 0),))

    # Create PerspectiveCameras
    cameras = PerspectiveCameras(device='cpu', R=R, T=T)

    # Define rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=128,
        radius=0.01,
        points_per_pixel=1,
    )

    # Create the rasterizer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Rasterize the point cloud
    fragments = rasterizer(point_cloud)
    # Get the indices of the nearest points
    idx = fragments.idx.squeeze(-1)  # Shape: (batch_size, H, W)

    # Get the depth values
    depth = fragments.zbuf.squeeze(-1)  # Shape: (batch_size, H, W)

    # Initialize the output tensor
    batch_size, H, W = idx.shape
    output = torch.zeros((batch_size, H, W, 4))  # 4 channels: X, Y, Z, depth

    for b in range(batch_size):
        for i in range(H):
            for j in range(W):
                point_idx = idx[b, i, j]
                if point_idx >= 0:
                    output[b, i, j, :3] = points[point_idx]
                    output[b, i, j, 3] = depth[b, i, j]
    
    return output

def main():
    path = "meshes\\66.obj"
    mesh = o3d.io.read_triangle_mesh(path)
    points = mesh.sample_points_uniformly(number_of_points=n)
    points = torch.tensor(points,dtype=torch.float32)

    projs = pinhole_camera_projection(points)
    ts.show(projs[0,:,:3]) # original xyz coords
    ts.show(projs[0,:,:,4]) # depth