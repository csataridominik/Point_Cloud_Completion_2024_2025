from tqdm import tqdm
from auxiliary_functions import generate_incomplete
import os
#from train import train
from project2 import project2,read_pcd
from project import create_sparse_input,load_point_cloud,project
import numpy as np

'''
- First load ships
- Make them incomplete:
    + Make 3-6 incomplete pcd's from 1 pcd
- Save these in a dataset with y being the originals projection and
X the incomplete projection's
- train the model in a training loop
'''


from raycasting_projections import raycasting_projection

def incomplete_input():
    sparse_dir = 'XYZ_projections_sparse\\'
    incomplete_pcd_dir = 'XYZ_projections_incomplete\\'
    min_idx = 0
    max_idx = 255

    for idx in tqdm(range(min_idx,max_idx)):
        path = sparse_dir + str(idx) + ".npy"
        all_images = np.load(path)
        coordinates = []
        epsilon = 0.1
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

        incomplete_pcd = generate_incomplete(coordinates,False)

    

'''
Changing the sparse:
- group .npy -> 2400 points for an object.
- group _1200.npy -> 1200 points for an object. Changed in raycasting_projections !!!
- etc (600 and 1800)
'''
def main4():
    path = 'meshes\\'
    projection_dir = 'XYZ_projections'
    incomplete_dir = 'XYZ_projections_sparse'
    
    obj_files = [f for f in os.listdir(path) if f.endswith('.obj')]
    

    min_idx = 65
    max_idx = 66
    for idx in tqdm(range(min_idx,max_idx)):

        mesh_path = path + str(idx) + ".obj" 

        output_dir = projection_dir

        #raycasting_projection(mesh_path,output_dir,incomplete_dir,idx,1200) # used be 2400
        raycasting_projection(mesh_path,output_dir,incomplete_dir,idx,800)
        #raycasting_projection(mesh_path,output_dir,incomplete_dir,idx,1800)
def main3():
    path = 'meshes\\'
    projection_dir = 'XYZ_projections'
    incomplete_dir = 'XYZ_projections_sparse'
    
    obj_files = [f for f in os.listdir(path) if f.endswith('.obj')]
    

    idx=0
    max_idx = 256
    for idx in tqdm(range(max_idx)):
        

        mesh_path = path + str(idx) + ".obj"

        output_dir = projection_dir

        project2(mesh_path,idx,output_dir,number_of_points=100_000) 

        output_dir = incomplete_dir
        project2(mesh_path,idx,output_dir,number_of_points=500)
        


# main2 is for projecting images to XYZ not to depth maps...
def main2():
    path = 'data'
    projection_dir = 'XYZ_projections'
    incomplete_dir = 'XYZ_projections_sparse'
    pcd_files = [f for f in os.listdir(path) if f.endswith('.pcd')]
    dataset_size = len(pcd_files)

    number_of_incomplete_samples = 1 # This many incomplete samples are generated from 1 ship
    for pcd_file in tqdm(pcd_files):
    
        pcd = load_point_cloud(os.path.join(path, pcd_file))
        idx = os.path.splitext(pcd_file)[0]
        output_dir = projection_dir

        pcd_coords = np.asarray(pcd.points)

        project2(pcd_coords,idx,output_dir) 

        incomlete_pcd = create_sparse_input(pcd,1000) # I use 1000 points here as input...
        pcd_coords = np.asarray(incomlete_pcd.points)
        output_dir = incomplete_dir
        project2(pcd_coords,idx,output_dir)


def main():
    path = 'data'
    projection_dir = 'projected_depth_maps'
    incomplete_dir = 'sparse_depth_maps'
    pcd_files = [f for f in os.listdir(path) if f.endswith('.pcd')]
    dataset_size = len(pcd_files)

    number_of_incomplete_samples = 1 # This many incomplete samples are generated from 1 ship
    for pcd_file in tqdm(pcd_files):
    
        pcd = load_point_cloud(os.path.join(path, pcd_file))
        output_dir = os.path.join(projection_dir, os.path.splitext(pcd_file)[0])

        project(pcd,output_dir) 

        for i in range(0,number_of_incomplete_samples):
            #incomlete_pcd = generate_incomplete(pcd,False)
            output_dir = os.path.join(incomplete_dir, os.path.splitext(pcd_file)[0]+f'_{i}')
       
        #    project(pcd,output_dir,800)

    #dataset = create_dataset(dataset_size,number_of_incomplete_samples,incomplete_dir="sparse_depth_maps")
    #train(dataset)

#main()
#main2()
#main3()
main4()
