from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle

from einops import rearrange


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
    
    train_size = int(train_ratio * len(dataset))
    '''
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    '''
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = 0
    return train_loader,test_loader

def load_dataloader(processed_idx):
    file_path = 'dataloaders/train' + str(processed_idx) + '.pkl'

    with open(file_path, 'rb') as file:
        dataloader_train = pickle.load(file)

    file_path = 'dataloaders/test' + str(processed_idx) + '.pkl'

    with open(file_path, 'rb') as file:
        dataloader_test = pickle.load(file)

    return dataloader_train, dataloader_test

def save_dataloader(dataloader,processed_idx,type):
    with open('dataloaders/'+type+str(processed_idx)+'.pkl', 'wb') as f:
        pickle.dump(dataloader, f)
    
def save_dataloader(y,processed_idx,type):
    with open('dataloaders/'+type+str(processed_idx)+'.pkl', 'wb') as f:
        pickle.dump(y, f)

def manage_dataloaders(dataset_size,number_of_chunks = 60,batch_size = 4):
    chunk_size = dataset_size // number_of_chunks

    for i in tqdm(range(number_of_chunks)):
        dataset = create_dataset_pinhole_projections(i*chunk_size,(i+1)*chunk_size)

        train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)
        save_dataloader(train_dataloader,i,"train")
        save_dataloader(test_dataloader,i,"test")

def create_dataset2(it,dataset_size,incomplete_dir = 'XYZ_projections_sparse\\XYZ_'):
    projection_dir = 'XYZ_projections\\XYZ_'
        
    X = []
    y = []

    for i in range(it,dataset_size):
        GT_XYZ = np.load(projection_dir+str(i)+".npy")
        y.append(GT_XYZ) 

        input_XYZ = np.load(incomplete_dir+str(i)+".npy")
        X.append(0)# X.append(input_XYZ) !!!!!!!!!!!!!!!!!!its changed------------------------------------------------- 
    
    #X = torch.from_numpy(X).float()
    #y = torch.tensor(np.asarray(y,dtype=np.float32),dtype=torch.float32)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset

def create_dataset_mixed_density(it,dataset_size,projection_dir = 'XYZ_projections\\XYZ_',incomplete_dir = 'XYZ_projections_sparse\\XYZ_',dropout=0.0):

    X = []
    y = []

    for i in range(it,dataset_size):
        for j in range(4):
            if j == 0:
                GT_XYZ = np.load(projection_dir+str(i)+".npy")
                input_XYZ = np.load(incomplete_dir+str(i)+".npy")
            elif j == 1:
                GT_XYZ = np.load(projection_dir+str(i)+"_1800.npy")
                input_XYZ = np.load(incomplete_dir+str(i)+"_1800.npy")
            elif j == 2:
                GT_XYZ = np.load(projection_dir+str(i)+"_1200.npy")
                input_XYZ = np.load(incomplete_dir+str(i)+"_1200.npy")
            elif j == 3:
                GT_XYZ = np.load(projection_dir+str(i)+"_800.npy")
                input_XYZ = np.load(incomplete_dir+str(i)+"_800.npy")
            
            
            if np.random.rand() <= dropout:
                view = np.random.randint(0,4)
                input_XYZ[:,:,view] = -1 # broadcasts... masks out one of the inputs
            
            y.append(GT_XYZ)
            X.append(input_XYZ) 


    
    #X = torch.from_numpy(X).float()
    #y = torch.tensor(np.asarray(y,dtype=np.float32),dtype=torch.float32)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset

def create_dataset_refinment(it,dataset_size,projection_dir = 'XYZ_projections\\XYZ_',incomplete_dir = 'DDIM_predictions\\',dropout=0.0):

    X = []
    y = []

    for i in range(it,dataset_size):
        GT_XYZ = np.load(projection_dir+str(i)+".npy")
        
        GT_XYZ[2:4,:,:,:] = np.flip(GT_XYZ[2:4,:,:,:],axis=2)
        # v,h,w,c -> c,h*v/2,w*v/2
        GT_XYZ = rearrange(GT_XYZ, '(b1 b2) h w c -> c (b1 h) (b2 w)', b1=2, b2=2)
        
        input_XYZ = np.load(incomplete_dir+str(i)+".npy")


        y.append(GT_XYZ)
        X.append(input_XYZ) 


    
    #X = torch.from_numpy(X).float()
    #y = torch.tensor(np.asarray(y,dtype=np.float32),dtype=torch.float32)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset

def create_dataset_pinhole_projections(it,dataset_size,dir = 'wsl_prep_data'):

    X = []
    y = []

    for i in range(it,dataset_size):
        GT_XYZ = np.load(os.path.join(dir, "target", f'target_{i}.npy'))

        placeholder = GT_XYZ[2,:,:,:].copy()
        GT_XYZ[2,:,:,:] = GT_XYZ[3,:,:,:]
        GT_XYZ[3,:,:,:] = placeholder
        GT_XYZ[2:4,:,:,:] = np.flip(GT_XYZ[2:4,:,:,:],axis=2)
        
        incomplete = np.load(os.path.join(dir, "incomplete_projections", f'{i}.npy'))
        placeholder = incomplete[2,:,:,:].copy()
        incomplete[2,:,:,:] = incomplete[3,:,:,:]
        incomplete[3,:,:,:] = placeholder
        incomplete[2:4,:,:,:] = np.flip(incomplete[2:4,:,:,:],axis=2)
        
        y.append(GT_XYZ) 

        X.append(incomplete)
        
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))


    dataset = IncompleteDataset(X, y)
    
    return dataset 

#-------------------------------
