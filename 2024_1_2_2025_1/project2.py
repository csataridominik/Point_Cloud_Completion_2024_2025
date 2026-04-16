# The functions projection_to_plane() and interpolate_coordinates_to_2D_grid() was highly cotributed and adviced by: Dr. Hatvani Janka | 2024.11.22.
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from PIL import Image
from reproject2 import reproject2


def filter_visible_points(pcd, coordinates_on_plane, mesh):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Convert open3d.geometry.TriangleMesh to open3d.t.geometry.TriangleMesh
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a raycasting scene and add the triangle mesh
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = raycasting_scene.add_triangles(t_mesh)

    visible_points = []
    
    # Prepare rays for each point in the point cloud
    rays = []
    for idx,point in enumerate(pcd):  # Assuming pcd is an open3d.geometry.PointCloud
        
        ray_origin = coordinates_on_plane[idx]
        ray_direction = point - ray_origin
        ray_direction /= np.linalg.norm(ray_direction)  # Normalize
        
        # Append the ray (origin + direction) to the rays list
        rays.append(np.hstack((ray_origin, ray_direction)))

    # Convert rays to tensor (Nx6 shape)
    rays_tensor = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)

    # Perform raycasting
    hits = raycasting_scene.cast_rays(rays_tensor)
    epsilon = 1e-4  # Small threshold to handle points close to the mesh
    

    for i, hit in enumerate(hits['geometry_ids'].numpy()):
        
        if hit != mesh_id:  # If the ray doesn't hit the mesh before reaching the point

            visible_points.append(pcd[i])
        else:
            # The ray hit the mesh, check if the hit distance is smaller than the distance to the point
            hit_distance = hits['t_hit'].numpy()[i]
            point_distance = np.linalg.norm(coordinates_on_plane[i] - pcd[i])

            # If the hit distance is greater than the point's distance, the point is visible
            if abs(hit_distance-point_distance) <= epsilon:
                visible_points.append(pcd[i])

    return np.array(visible_points)



def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError("The point cloud is empty or could not be loaded.")
    
    pcd = np.asarray(pcd.points)

    return pcd


def projection_to_plane(points, plane_normal, plane_origin):
    """
    Projects the given points to the plane defined by the normal and the origin.

    Args:
        points (np.ndarray): The points to be projected.
        plane_normal (np.ndarray): The normal of the plane.
        plane_origin (np.ndarray): The origin of the plane.

    Returns:
        coordinates_on_plane (np.ndarray): The 3D coordinates of the points on the plane.

    """
    # normalize the normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # calculate the distance of the plane from the origin
    d = plane_normal.dot(plane_origin)

    # calculate the distance of the points from the plane
    t = (d - np.dot(points, plane_normal)) / np.dot(plane_normal, plane_normal)
    # calculate the coordinates of the points on the plane
    coordinates_on_plane = points + np.expand_dims(t, axis=1) * np.expand_dims(plane_normal, axis=0)
    return coordinates_on_plane 

def interpolate_coordinates_to_2D_grid(coordinates_3D, point_cloud, plane_normal, plane_origin, resolution):
    # rotate plane to xy plane
        # Rotate the coordinates from the plane to the xy plane
        z = plane_normal
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        if np.allclose(z, x) or np.allclose(z, -x):
            x = np.array([0, 1, 0])
            y = np.array([0, 0, 1])
        if np.allclose(z, y) or np.allclose(z, -y):
            x = np.array([1, 0, 0])
            y = np.array([0, 0, 1])
        x = x - np.dot(x, z) * z
        x = x / np.linalg.norm(x)
        y = y - np.dot(y, z) * z - np.dot(y, x) * x
        y = y / np.linalg.norm(y)
        R = np.vstack((x, y, z))
        coordinates_2D =  np.dot(coordinates_3D, R.T)

        x_min = np.min(coordinates_2D[:, 0])
        x_max = np.max(coordinates_2D[:, 0])
        y_min = np.min(coordinates_2D[:, 1])
        y_max = np.max(coordinates_2D[:, 1])

        x_len = int(np.ceil((x_max-x_min)/resolution+1))
        y_len = int(np.ceil((y_max-y_min)/resolution+1))

        X = np.linspace(x_min, x_max, x_len)
        Y = np.linspace(y_min, y_max, y_len)
        X, Y = np.meshgrid(X, Y, indexing='ij')

        output = np.ones((x_len, y_len))*(-1)

        for coor, ind in zip(coordinates_2D, range(len(coordinates_2D))):
            x_idx = int(np.floor((coor[0]-x_min)/resolution))
            y_idx = int(np.floor((coor[1]-y_min)/resolution))
            if output[x_idx, y_idx] == -1:
                output[x_idx, y_idx] = ind

        mask = output != -1
        output_x = np.zeros((x_len, y_len))
        output_y = np.zeros((x_len, y_len))
        output_z = np.zeros((x_len, y_len))
        output_x[mask] = point_cloud[output[mask].flatten().astype(int)][:, 0]
        output_y[mask] = point_cloud[output[mask].flatten().astype(int)][:, 1]
        output_z[mask] = point_cloud[output[mask].flatten().astype(int)][:, 2]

        return output_x, output_y, output_z

def interpolate_coordinates_to_2D_grid2(coordinates_3D, point_cloud, plane_normal, plane_origin, resolution):
    
    distances = np.linalg.norm(point_cloud - coordinates_3D, axis=1)
    
    # Sort the point_cloud and coordinates_3D based on the distances
    sorted_indices = np.argsort(distances)
    point_cloud = point_cloud[sorted_indices]
    coordinates_3D = coordinates_3D[sorted_indices]
    
    # rotate plane to xy plane
        # Rotate the coordinates from the plane to the xy plane
    z = plane_normal
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    if np.allclose(z, x) or np.allclose(z, -x):
        x = np.array([0, 1, 0])
        y = np.array([0, 0, 1])
    if np.allclose(z, y) or np.allclose(z, -y):
        x = np.array([1, 0, 0])
        y = np.array([0, 0, 1])
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)
    y = y - np.dot(y, z) * z - np.dot(y, x) * x
    y = y / np.linalg.norm(y)
    R = np.vstack((x, y, z))
    coordinates_2D =  np.dot(coordinates_3D, R.T)

    x_min = np.min(coordinates_2D[:, 0])
    x_max = np.max(coordinates_2D[:, 0])
    y_min = np.min(coordinates_2D[:, 1])
    y_max = np.max(coordinates_2D[:, 1])

    x_len = int(np.ceil((x_max-x_min)/resolution+1))
    y_len = int(np.ceil((y_max-y_min)/resolution+1))

    X = np.linspace(x_min, x_max, x_len)
    Y = np.linspace(y_min, y_max, y_len)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    output = np.ones((x_len, y_len))*(-1)
    n = 0
    for coor, ind in zip(coordinates_2D, range(len(coordinates_2D))):
        x_idx = int(np.floor((coor[0]-x_min)/resolution))
        y_idx = int(np.floor((coor[1]-y_min)/resolution))

        neighberhood = 1
        alpha = 0.001
        
        if output[x_idx, y_idx] == -1:
            indecies = output[max(0,x_idx-neighberhood):min(x_idx+neighberhood+1,x_len):, max(0,y_idx-1):min(x_len,y_idx +neighberhood+1)]
            if np.all(indecies < 0):
                output[x_idx, y_idx] = ind
            else:
                for curr_index in indecies.flatten():
                    if curr_index != -1:
                        if np.linalg.norm(point_cloud[ind] - coordinates_3D[ind]) <= \
                        np.linalg.norm(point_cloud[int(curr_index)] - coordinates_3D[int(curr_index)])+ alpha:
                            output[x_idx, y_idx] = ind
                        else:
                            n+=1
        else:
            if np.linalg.norm(point_cloud[ind] - coordinates_3D[ind]) < \
                np.linalg.norm(point_cloud[int(output[x_idx, y_idx])] - coordinates_3D[int(output[x_idx, y_idx])]): 
                output[x_idx, y_idx] = ind

    mask = output != -1
    output_x = np.zeros((x_len, y_len))
    output_y = np.zeros((x_len, y_len))
    output_z = np.zeros((x_len, y_len))
    output_x[mask] = point_cloud[output[mask].flatten().astype(int)][:, 0]
    output_y[mask] = point_cloud[output[mask].flatten().astype(int)][:, 1]
    output_z[mask] = point_cloud[output[mask].flatten().astype(int)][:, 2]
    #print(f'This many are thrown out: {n}.')
    return output_x, output_y, output_z

def plotter(cloud,coordinates_on_plane):
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(cloud), color='blue', point_size=5)
    plotter.add_mesh(pv.PolyData(coordinates_on_plane), color='red', point_size=5)

    plotter.show()

def plot_interpol(x,y,z):
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(x)
    axes[1].imshow(y)
    axes[2].imshow(z)

    plt.show()

def save_image(XYZ,name):
    R = (255 * (XYZ / np.max(XYZ))).astype(np.uint8)
    R = np.transpose(R, (1, 2, 0))
    img = Image.fromarray(R, mode='RGB')  
    img.save(name)


def project2(mesh_path,idx,output_dir,number_of_points=100_000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pcd = np.asarray(pcd.points)

    views = 4
    all_images = []

    theta = np.radians(45)  # Angle of inclination from vertical
    r = 1.0  # Distance of cameras from the global origin
    global_origin = np.array([0.0, 0.0, 0.0])  # The point cloud's origin

    angles = [0, 90, 180, 270]  # Azimuth angles for cameras in degrees
    camera_positions = []

    W,H = 100,100

    for view in range(views):
        
        '''
        phi = np.radians(angle)  # Convert azimuth angle to radians
        x = r * np.cos(phi)      # X-coordinate in the plane
        y = r * np.tan(theta)      # Y-coordinate in the plane
        z = r * np.sin(phi)    # Height based on the inclination angle
        '''

        if view == 0:
            x = 1.0
            y = 1.0
            z = 1.0
        elif view ==1:
            x = -1.0
            y = 1.0
            z = 1.0
        elif view == 2:
            x = 1.0
            y = 1.0
            z = -1.0
        else:
            x = -1.0
            y = 1.0
            z = -1.0

        camera_positions.append(np.array([x, y, z]))

    # Compute camera directions (pointing toward the global origin)
    camera_directions = []
    for pos in camera_positions:
        direction = global_origin - pos  # Vector pointing to the origin
        direction = direction / np.linalg.norm(direction)  # Normalize the direction
        camera_directions.append(direction)

    for view in range(views):
        coordinates_on_plane = projection_to_plane(pcd, camera_directions[view], camera_positions[view])
        pcd_modified = filter_visible_points(pcd,coordinates_on_plane,mesh)
        coordinates_on_plane = projection_to_plane(pcd_modified, camera_directions[view], camera_positions[view])
        x,y,z = interpolate_coordinates_to_2D_grid(coordinates_on_plane, pcd_modified, camera_directions[view], camera_positions[view], 0.01)
        
        #print(f"This is size of x:  {x.shape}")
        #plot_interpol(x,y,z)
        XYZ = np.stack((x, y, z), axis=-1)
        
        n = W - x.shape[1]
        m = H - x.shape[0]

        if n < 0 or m < 0:
            print(f'The shape is too big so it gets disregarded...')
        else:
            padded_XYZ = np.pad(XYZ, ((m//2, m-m//2), (n//2, n-n//2), (0, 0)), mode='constant', constant_values=0)

        # This part is only here if you want to save it image form converted to RGB:
        #name = "XYZ_blobs"+str(view)+".png"
        #save_image(padded_XYZ,name)



        mask = np.any(padded_XYZ == 0, axis=-1)
        
        padded_XYZ = (padded_XYZ + 1) / 2 # normalize adter we know the background, than do the masking afterwards...
        # Set the background to -1 wherever the mask is True
        padded_XYZ[mask] = 0
        all_images.append(padded_XYZ)

        """
        from predict import plot_
        plot_(padded_XYZ.transpose(2,0,1)[0],"asd")
        plot_(padded_XYZ.transpose(2,0,1)[1],"asd")
        plot_(padded_XYZ.transpose(2,0,1)[2],"asd")
        """
    
    # Saving the file:
    all_images = np.stack(all_images, axis=0)
    save_path = output_dir + "\XYZ_" + str(idx) + ".npy"
    np.save(save_path, all_images)



#path = "data/65.pcd"
#pcd = read_pcd(path)
#project2(pcd,0,'XYZ_projections')


