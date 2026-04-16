import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from tqdm import tqdm
import os
import fpsample


def load_point_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    if not pcd.has_points():
        raise ValueError("The point cloud is empty or could not be loaded.")
    return pcd

def generate_depth_image(idx,pcd, viewpoint,rotation_angles, width=640, height=480):
    # Create visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)  # Set window size
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Configure viewpoint (extrinsic camera parameters)
    view_control = vis.get_view_control()
    view_control.set_lookat(viewpoint['lookat'])
    view_control.set_front(viewpoint['front'])
    view_control.set_up(viewpoint['up'])
    view_control.rotate(rotation_angles[0], rotation_angles[1], rotation_angles[2],rotation_angles[3])
    
    # Define and apply camera intrinsics and extrinsic:
    parameters = view_control.convert_to_pinhole_camera_parameters()
    extrinsic_matrix = np.asarray(parameters.extrinsic)
    intrinsic_matrix = np.asarray(parameters.intrinsic.intrinsic_matrix)
    

    # Poll events and update the renderer
    vis.poll_events()
    vis.update_renderer()

    # Capture depth image
    depth_image = vis.capture_depth_float_buffer()

    # Clean up and destroy the window
    vis.destroy_window()

    # Convert depth image to NumPy array
    depth_image_np = np.asarray(depth_image)

    return depth_image_np,extrinsic_matrix,intrinsic_matrix

def create_sparse_input(pcd,numberofpoints):
    
    pcd_o3d = o3d.geometry.PointCloud()
    
    pcd_coords = np.asarray(pcd.points)
    
    sampled = fpsample.bucket_fps_kdline_sampling(pcd_coords, numberofpoints, h=7) # h is recommended for medium data
    
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_coords[sampled])
    
    return pcd_o3d

def project(pcd,output_dir, sparse = 0):
    
    if sparse != 0:
        pcd = create_sparse_input(pcd, sparse)

    # Viewpoints for looking at the ship
    viewpoints = [
        {'lookat': [0, 0, 0], 'front': [1, 1, 0], 'up': [0, 1, 0]},  # Looking from +X axis 
    ]

    # Process each viewpoint and generate images
    for i in range(4):
        # Set rotation angles based on viewpoint index
        if i == 0:
            depth_image,extrinsic,intrinsic = generate_depth_image(i, pcd, viewpoints[0], [200, 1, 1, 1])
        elif i == 1:
            depth_image,extrinsic,intrinsic = generate_depth_image(i, pcd, viewpoints[0], [-200, -100, 1, 1])
        elif i == 2:
            depth_image,extrinsic,intrinsic = generate_depth_image(i, pcd, viewpoints[0], [-750, 150, 0, 0])
        else:
            depth_image,extrinsic,intrinsic = generate_depth_image(i, pcd, viewpoints[0], [820, 200, 0, 0])

        camera = os.path.join(output_dir, 'camera')
        os.makedirs(camera, exist_ok=True)

        np.savetxt(camera + f'/extrinsic{i}.txt', extrinsic, fmt='%.10f')  
        if i == 0:
            np.savetxt(camera + f'/intrinsic.txt', intrinsic, fmt='%.10f')  

        # Normalize depth image to the range [0, 255]
        depth_image_normalized = (255 * (depth_image - np.min(depth_image)) / np.ptp(depth_image)).astype(np.uint8)
        
        # Create a PIL Image from the numpy array
        #image = Image.fromarray(depth_image_normalized)

    
        # Here I change all the 0 values to -1: 
        
        depth_image[depth_image == 0] = -1
        # ----

        '''
        # Display depth map visualization
        plt.imshow(depth_image, aspect='auto')
        plt.axis('off')
        plt.title('Depth Map Visualization')
        plt.show()'''
        
        # Save the PNG and TXT for the depth image
        # png_filename = os.path.join(output_dir, f'depth_image_{i}.png')
        txt_filename = os.path.join(output_dir, f'depth_image_{i}.txt')
        
        # Save as image
        # image.save(png_filename)

        # Save as text
        np.savetxt(txt_filename, depth_image)
        

#project('data','projected_depth_maps')
