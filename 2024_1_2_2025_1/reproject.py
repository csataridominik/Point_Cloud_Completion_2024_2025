import open3d as o3d
import numpy as np
from PIL import Image

def reproject(path):
    # Load depth images
    depth_images = [
        np.loadtxt(path + 'depth_image_0.txt'),
        np.loadtxt(path + 'depth_image_1.txt'),
        np.loadtxt(path + 'depth_image_2.txt'),
        np.loadtxt(path + 'depth_image_3.txt')
    ]

    pcd = []
    pcd2 = []
    # Select the first depth image
    for i in range(4):
        depth = depth_images[i]

        # Convert depth image to float32 and scale it (assuming depth is in mm)
        depth_float = depth.astype(np.float32)  # Convert to meters
        intrinsic_matrix = np.loadtxt(path + 'camera/intrinsic.txt')
        
        # Assuming you have the image width and height from the camera
        width = 640  # Example width (replace with actual value)
        height = 480  # Example height (replace with actual value)

        # Extract focal lengths and principal point from the intrinsic matrix
        f_x = intrinsic_matrix[0, 0]
        f_y = intrinsic_matrix[1, 1]
        c_x = intrinsic_matrix[0, 2]
        c_y = intrinsic_matrix[1, 2]

        # Create the PinholeCameraIntrinsic object
        intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width, height, f_x, f_y, c_x, c_y)
        intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width, height, f_x, f_y, c_x, c_y)
        
        depth_o3d = o3d.geometry.Image(depth_float)

        if i ==0:
            extrinsic = np.loadtxt(path + 'camera/extrinsic0.txt')
            pcd1 = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic1,extrinsic,depth_scale=10.0)
        elif i ==1:
            extrinsic = np.loadtxt(path + 'camera/extrinsic1.txt')
            pcd2 = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic2,extrinsic,depth_scale=10.0)
        elif i == 2:
            extrinsic = np.loadtxt(path + 'camera/extrinsic2.txt')
            pcd3 = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic1,extrinsic,depth_scale=10.0)
        elif i == 3:
            extrinsic = np.loadtxt(path + 'camera/extrinsic3.txt')
            pcd4 = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic2,extrinsic,depth_scale=10.0)


    # Optionally, visualize the point cloud
    o3d.visualization.draw_geometries([pcd1 + pcd2 + pcd3 + pcd4])

