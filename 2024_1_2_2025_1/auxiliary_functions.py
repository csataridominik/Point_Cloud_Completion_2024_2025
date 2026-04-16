import numpy as np
import open3d as o3d
import random

def generate_incomplete_by_plane_cut(points, plane_normal, plane_point,keep_ratio=0.51):
    # Subtract plane_point from all points
    vec_to_points = points - plane_point

    number_of_points = points.shape[0]
    print(number_of_points)

    # Calculate distances from the plane
    distances = np.sum(vec_to_points * plane_normal, axis=-1)

    mask = distances >= 0

    #print(f'This is mask sum: {np.sum(mask)}')
    
    if np.sum(mask)/number_of_points <= keep_ratio:
        mask = distances <= 0

    # Filter points that meet the criteria
    filtered_points = points[mask]

    return filtered_points


# This is for visualisation:
def display_open3d(input_pc, output):
    input_pc_ = o3d.geometry.PointCloud()
    output_ = o3d.geometry.PointCloud()

    input_pc_.points = o3d.utility.Vector3dVector(input_pc)
    output_.points = o3d.utility.Vector3dVector(output + np.array([1,0,0]))
    #input_pc_.paint_uniform_color([1, 0, 0]) # This is red probably for input
    #output_.paint_uniform_color([0, 1, 0]) # This is the green for output
    o3d.visualization.draw_geometries([input_pc_, output_])

def generate_incomplete_by_boundingbox(pcd,center=[0, 0, 0],extent=[1, 1, 1]):
    bounding_box = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3), extent=extent)

    #cropped_pcd = pcd.crop(bounding_box)
    inliers_indices = bounding_box.get_point_indices_within_bounding_box(pcd.points)
 
    incomplete_pcd = pcd.select_by_index(inliers_indices, invert=True)

    return incomplete_pcd

def generate_incomplete(pcd,cut=False):

    vertices = np.asarray(pcd.points)
    random_index = random.randint(0, len(vertices) - 1)  # get random index

    if cut:        
        plane_normal = np.array([1, 0.5, 0], dtype=np.float32)
        plane_point = np.mean(vertices, axis=0) + [0.05, 0.05, 0.05]
        filtered_points = generate_incomplete_by_plane_cut(vertices, plane_normal, plane_point)
        incomplete_pcd = o3d.geometry.PointCloud()
        incomplete_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    else:
        incomplete_pcd = generate_incomplete_by_boundingbox(pcd, vertices[random_index][:],[random.uniform(0.15, 0.3),random.uniform(0.15, 0.3),random.uniform(0.15, 0.3)])

    # This part is for displaying the results:

    #display_open3d(vertices,filtered_points)
    #display_open3d(vertices,np.asarray(incomplete_pcd.points))

    return incomplete_pcd

