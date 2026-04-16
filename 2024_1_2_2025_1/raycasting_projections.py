# The function ray_casting_to_coordinates_from_origin() and closest_point_coordinates() was highly cotributed and adviced by: Dr. Hatvani Janka | 2024.11.30.
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt

def read_pcd(path):
    cloud = pv.read(path)
    # points = cloud.points
    # faces = cloud.faces
    return cloud

def ray_casting_to_coordinates_from_origin(mesh, target_coordinates, origin):
    """
    Casts rays from the origin to the target coordinates on the mesh and returns the coordinates where the rays intersect the mesh.

    Args:
        mesh (pyvista.PolyData): The mesh to cast rays on.
        target_coordinates (np.ndarray): The coordinates to cast rays to.
        origin (np.ndarray): The origin of the rays.

    Returns:
        intersection_coordinates (np.ndarray): The coordinates where the rays intersect the mesh.

    """
    # Calculate the direction of the rays
    # directions = target_coordinates - origin
    # directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    # Cast the rays
    intersection_coordinates = np.ones(target_coordinates.shape)*(-1)
    for i in range(target_coordinates.shape[0]):
        intersection = mesh.ray_trace(origin[i], target_coordinates[i],first_point = True,plot=False)[0]
        if intersection.size != 0:
            intersection_coordinates[i] = intersection
    return intersection_coordinates

from scipy.spatial.distance import cdist
'''
def distance_image(target_coordinates, pcd_coordinates):

    # distance matrix
    D = cdist(target_coordinates, pcd_coordinates, metric='euclidean')

    closest_indices = np.argmin(D, axis=0)  
    mask = np.zeros_like(D, dtype=bool)
    mask[closest_indices, np.arange(D.shape[1])] = True

    D_nan = np.full_like(D, np.nan, dtype=float)
    D_nan[mask] = D[mask]
    
    output = np.ones(target_coordinates.shape)*(-1)
    # for each target point we select from the remaining pcd points the closest
    D[]=nan

    return output
'''


def select_closest_points(target_coordinates, pcd_coordinates):
    
    D = cdist(target_coordinates,pcd_coordinates, metric='euclidean')
    

    closest_idx = np.argmin(D,axis=1)
    selected_coords = []
    print(f'This is shape of pcd: {pcd_coordinates.shape}')
    print(closest_idx.shape[0])
    output = np.ones(target_coordinates.shape)*(-1)

    #for i in range(closest_idx.shape[0]):
        
    output[closest_idx] = target_coordinates[closest_idx]
        #selected_coords.append(target_coordinates[closest_idx[curr_x]]) 
    
    return output


def closest_point_coordinates(target_coordinates, pcd_coordinates):
    # Compute the distance matrix (Euclidean distances between target and pcd points)
    D = cdist(target_coordinates,pcd_coordinates, metric='euclidean')

    # Find the closest point in the pcd for each target point
    #closest_indices = np.argmin(D, axis=1)  # Axis=1 because we're looking for the closest pcd point for each target point

    '''
    Here:
    loop thru each closest indices, and have a D_nan matrix set
    to all false, but when we select an index from pcd_coords
    put the whole row of it in d_nan to True.
    If the value in d_nan is true for index pcd_coord than we
    append selected_coords([-1,-1,-1])
    '''
    output = np.ones(target_coordinates.shape)*(-1)
    selected_coords = np.ones(pcd_coordinates.shape[0])*(0)

    for idx in range(target_coordinates.shape[0]):
        
        closest_idx = np.argmin(D[idx])
        if selected_coords[closest_idx] == 0:
            output[idx] = pcd_coordinates[closest_idx]
            selected_coords[closest_idx] = 1
        else:
            output[idx] = [-1,-1,-1]


    return output

import numpy as np
from scipy.spatial.distance import cdist



def rotate_2D_coordinates_to_3D(coordinates_2D, plane_normal, plane_origin):
    """
    Rotates the 2D coordinates to 3D coordinates on the plane defined by the normal and the origin.
    The (0,0,0) coordinate of the 2D coordinates should be the origin of the plane.
    

    """
    # rotate the 2d coordinates with the plane normal
    normal_2d = np.array([0,0,1])
    rotation_matrix = rotation_matrix_from_normals(normal_2d, plane_normal)
    coordinates_3D = np.dot(coordinates_2D, rotation_matrix.T)
    # translate the 2d coordinates to the plane
    coordinates_3D = coordinates_3D + plane_origin
    return coordinates_3D

import numpy as np

def rotation_matrix_from_normals(normal1, normal2):
    # Normalize the input vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)
    
    # Calculate the axis of rotation (cross product of the two normals)
    axis = np.cross(normal1, normal2)
    axis_length = np.linalg.norm(axis)
    
    # If the axis length is zero, the vectors are parallel or anti-parallel
    if axis_length == 0:
        if np.dot(normal1, normal2) > 0:
            return np.eye(3)  # No rotation needed
        else:
            # 180 degree rotation around any orthogonal axis
            orthogonal_axis = np.array([1, 0, 0]) if abs(normal1[0]) < abs(normal1[1]) else np.array([0, 1, 0])
            axis = np.cross(normal1, orthogonal_axis)
            axis = axis / np.linalg.norm(axis)
            axis_length = np.linalg.norm(axis)
    
    axis = axis / axis_length
    
    # Calculate the angle of rotation (dot product of the two normals)
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    
    # Compute the rotation matrix using the Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return rotation_matrix



def grid_2D(width,height):
    # Create a 2D grid
    x = np.linspace(-width/2, width/2, width)
    y = np.linspace(-height/2, height/2, height)
    X, Y = np.meshgrid(x, y)
    coordinates_2D = np.vstack((X.flatten(), Y.flatten(), np.zeros(X.flatten().shape))).T
    return coordinates_2D


def convert(o3d_mesh):

    o3d_mesh.compute_vertex_normals()

    vertices = np.asarray(o3d_mesh.vertices)  # Convert to NumPy array
    triangles = np.asarray(o3d_mesh.triangles)  # Convert to NumPy array

    # Create a PyVista mesh
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(triangles), 1), 3), triangles]))
    return pv_mesh



import open3d as o3d


'''
def raycasting_projections(path,n=500):
    path = "model_normalized.obj"
    mesh = o3d.io.read_triangle_mesh(path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=n)

    # Check the mesh before simplification
    print(f"Original mesh has {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices.")

    # Simplify the mesh using Quadric Decimation
    target_triangle_count = len(mesh.triangles) // 50  
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
    print(f"Simplified mesh has {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices.")

    mesh = convert(mesh)


    # plot the 3D coordinates, the origin and the mesh
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(coordinates_3D[:,0], coordinates_3D[:,1], coordinates_3D[:,2])
    #ax.scatter(origin_coordinates_3D[:,0], origin_coordinates_3D[:,1], origin_coordinates_3D[:,2])
    #ax.scatter(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2])
    #plt.show()
    


    #output_sparse[mask] = output[mask]

    print(output.shape)
    # -------------------------------------------------------------------------------------
    plot_(output, output_sparse)


'''


def plot_(output,output_sparse,width=128,height=128):
    output_x = np.reshape(output[:,0], (width,height))
    output_y = np.reshape(output[:,1], (width,height))
    output_z = np.reshape(output[:,2], (width,height))

    output_x_sparse = np.reshape(output_sparse[:,0], (width,height))
    output_y_sparse = np.reshape(output_sparse[:,1], (width,height))
    output_z_sparse = np.reshape(output_sparse[:,2], (width,height))


    fig, axs = plt.subplots(2, 3)

    # First row (index 1, 2, 3)
    axs[0, 0].imshow(output_x)      # First subplot
    axs[0, 1].imshow(output_y)      # Second subplot
    axs[0, 2].imshow(output_z)      # Third subplot

    # Second row (index 4, 5, 6)
    axs[1, 0].imshow(output_x_sparse)   # Fourth subplot
    axs[1, 1].imshow(output_y_sparse)   # Fifth subplot
    axs[1, 2].imshow(output_z_sparse)   # Sixth subplot (last one)

    plt.show()

def fun(mesh, target_coordinates, origin):
    """
    Casts rays from the origin to the target coordinates on the mesh and returns the coordinates where the rays intersect the mesh.

    Args:
        mesh (pyvista.PolyData): The mesh to cast rays on.
        target_coordinates (np.ndarray): The coordinates to cast rays to.
        origin (np.ndarray): The origin of the rays.

    Returns:
        intersection_coordinates (np.ndarray): The coordinates where the rays intersect the mesh.

    """
    # Calculate the direction of the rays
    # directions = target_coordinates - origin
    # directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    # Cast the rays
    intersection_coordinates = np.ones(target_coordinates.shape)*(-1)
    for i in range(target_coordinates.shape[0]):
        intersection = origin - target_coordinates[i]
        if intersection.size != 0:
            intersection_coordinates[i] = intersection
    return intersection_coordinates



from reproject2 import reproject2

def raycasting_projection(path,output_dir,output_dir_sparse,idx,n=500):
    
    mesh = o3d.io.read_triangle_mesh(path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=n)

    # Check the mesh before simplification
    #print(f"Original mesh has {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices.")
    
    #This part is needed!!-----------------------------------------------------------------------------
    target_triangle_count = len(mesh.triangles) // 5
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)
    #This part is needed!!-----------------------------------------------------------------------------

    #print(f"Simplified mesh has {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices.")
    #o3d.visualization.draw_geometries([mesh])
    mesh = convert(mesh)

    width = 128
    height = 128 # I have changed it from 256!!
    image_grid  = grid_2D(width,height)/width
    origin_grid = grid_2D(width,height)/height
    #mesh.points = mesh.points - np.mean(mesh.points, axis=0)
    #mesh.points = mesh.points / (np.max(mesh.points) - np.min(mesh.points))

    views = 4

    all_images = []
    all_sparse_images = []
    for view in range(views):

        if view ==0:
            origin = np.asarray([2,2,2])
            plane_origin = np.array([2,2,-2])
            plane_origin = np.asarray([-2,-2,2])
            plane_normal = origin - plane_origin
            plane_normal = plane_normal / np.linalg.norm(plane_normal) 

        elif view ==2:
            origin = np.asarray([-2,2,2])
            plane_origin = np.asarray([2,-2,-2])
            plane_normal = origin - plane_origin
            plane_normal = plane_normal / np.linalg.norm(plane_normal) 

        else:
            origin = np.asarray([-2,2,-2])
            plane_origin = np.asarray([2,-2,2])
            plane_normal = origin - plane_origin
            plane_normal = plane_normal / np.linalg.norm(plane_normal) 


        origin_coordinates_3D = rotate_2D_coordinates_to_3D(origin_grid, plane_normal, origin)
        coordinates_3D = rotate_2D_coordinates_to_3D(image_grid, plane_normal, plane_origin)
        intersection_coordinates = ray_casting_to_coordinates_from_origin(mesh, coordinates_3D, origin_coordinates_3D)
        output = intersection_coordinates
        selected_points = np.asarray(point_cloud.points)
        mask = output == -1
        output_sparse = closest_point_coordinates(intersection_coordinates,selected_points)

        output_x = np.reshape(output[:,0], (width,height))
        output_y = np.reshape(output[:,1], (width,height))
        output_z = np.reshape(output[:,2], (width,height))

        output_x_sparse = np.reshape(output_sparse[:,0], (width,height))
        output_y_sparse = np.reshape(output_sparse[:,1], (width,height))
        output_z_sparse = np.reshape(output_sparse[:,2], (width,height))

        XYZ = np.stack((output_x, output_y, output_z), axis=-1)
        XYZ_sparse = np.stack((output_x_sparse, output_y_sparse, output_z_sparse), axis=-1)
        
        mask = np.any(XYZ == -1, axis=-1)
        
        XYZ = (XYZ + 1) / 2 # normalize adter we know the background, than do the masking afterwards...
        # Set the background to -1 wherever the mask is True
        XYZ[mask] = 0
        
        mask = np.any(XYZ_sparse == -1, axis=-1)
        
        XYZ_sparse = (XYZ_sparse + 1) / 2 # normalize adter we know the background, than do the masking afterwards...
        # Set the background to -1 wherever the mask is True
        XYZ_sparse[mask] = 0

        
        all_images.append(XYZ)
        all_sparse_images.append(XYZ_sparse)

        #plot_(output,output_sparse)

    #reproject2(all_images)
    #reproject2(all_sparse_images)
        
    # Saving the file:
    
    all_images = np.stack(all_images, axis=0)
    save_path = output_dir + "\\XYZ_" + str(idx) + "_" + str(n)+".npy"
    np.save(save_path, all_images)

    all_sparse_images = np.stack(all_sparse_images, axis=0)
    save_path = output_dir_sparse + "\\XYZ_" + str(idx) + "_" + str(n)+".npy"
    np.save(save_path, all_sparse_images)
    
   

#raycasting_projection('meshes/0.obj',output_dir=0,output_dir_sparse=0,idx=0,n=500)