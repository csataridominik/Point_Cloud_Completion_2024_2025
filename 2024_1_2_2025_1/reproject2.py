import numpy as np
import open3d as o3d


# I have changed thee coords anf the x, y, z = image[:,i, j] from x, y, z = image[i, j]
def reproject2(all_images):
    coordinates = []
    epsilon = 0.0
    count = 0
    for image in all_images:
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                x, y, z = image[:,i, j]
                
                # Only keep the point if not all coordinates are zero
                if abs(x) > epsilon and abs(y) > epsilon and abs(z) > epsilon:
                    count += 1
                    coordinates.append([x, y, z])
    
    coordinates = np.array(coordinates)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(coordinates)

    o3d.visualization.draw_geometries([pcd])



