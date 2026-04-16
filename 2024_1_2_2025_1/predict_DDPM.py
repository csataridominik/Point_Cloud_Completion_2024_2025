from model import MultiViewCompletionNet,MultiViewCompletionNet2
from MVPCC_Networks import InpaintGenerator
from reproject import reproject
from reproject2 import reproject2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from raycasting_projections import plot_ as plot_2
from scipy import ndimage
from train_DDPM import UNet

def plot_(matrix,name):
    plt.imshow(matrix)

    plt.colorbar()  # Optional: shows a color scale bar
    plt.title(name)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

def predict(model,device,input,target=0,IsThereTarget=False):
    loss = nn.MSELoss()
    model.eval()
    
    input = input.reshape(4 * 3, input.shape[2], input.shape[3])  # Reshape to (4*3, h, w)
    
    single_input = torch.tensor([input], dtype=torch.float32)
    
    if IsThereTarget:
        target = target(0, 3, 1, 2).reshape(4 * 3, target.shape[2], target.shape[3])  # Reshape to (4*3, h, w)
        single_target = torch.tensor([target], dtype=torch.float32)
    
    if torch.cuda.is_available():
        single_input = single_input.to(device)

    with torch.no_grad():
        
        output = model(single_input)
    
    if IsThereTarget:
        loss = loss(output[0], single_target[0])
        return output, loss
    
    return output

def predict_main2(name,idx = "66"):
    target_path = "XYZ_projections\XYZ_"+idx+".npy"
    sparse_input_path = "XYZ_projections_sparse\XYZ_"+idx+".npy"

    target= np.load(target_path)
    target = target.transpose(0, 3, 1, 2).copy()

    sparse_input = np.load(sparse_input_path)
    sparse_input = sparse_input.transpose(0, 3, 1, 2).copy()


    path = "trained_model/"+name+".pth"
    model = UNet(in_channels=12)
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')),strict=False)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model

    
    output = predict(model,device,sparse_input,target,False)
    output = output.cpu().detach().numpy().squeeze()  # Convert to NumPy array
    #print(f'This is size of output: {output.shape}')
    
    output = output.reshape(4, 3, 256, 256)
    


    for i in range(3):
        for j in range(4): # (Only showing the first channel):
            
            plot_(sparse_input[i][j],"Input")
            plot_(target[i][j],"Target")
            plot_(output[i][j],"Output")
            break
    
    return output,sparse_input

#predict_main3()
#predict_main2("20250312_DiffusionModel_01","66")

