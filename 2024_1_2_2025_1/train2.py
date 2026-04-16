from model import MultiViewCompletionNet2
from MVPCC_Networks import InpaintGenerator
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import create_dataset2
from project2 import save_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from scipy import ndimage

import torch
import torch.nn.functional as F

def sobel_filter():
    """Creates Sobel filters for computing horizontal and vertical gradients."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return sobel_x, sobel_y

def compute_sobel_magnitude(tensor):
    """
    Compute Sobel magnitude for each channel separately.
    
    Args:
        tensor: Input tensor of shape [batch, 4, 3, 256, 256]
        
    Returns:
        sobel_magnitudes: Tensor of the same shape containing Sobel magnitudes.
    """
    
    sobel_x, sobel_y = sobel_filter()
    sobel_x, sobel_y = sobel_x, sobel_y

    batch, projections, channels, height, width = tensor.shape
    sobel_magnitudes = torch.zeros_like(tensor)

    for c in range(channels):  # Process x, y, z separately
        channel_data = tensor[:, :, c, :, :].reshape(-1, 1, height, width)  # Flatten batch & projections
        
        grad_x = F.conv2d(channel_data.cpu().detach(), sobel_x, padding=1)
        grad_y = F.conv2d(channel_data.cpu().detach(), sobel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        sobel_magnitudes[:, :, c, :, :] = magnitude.view(batch, projections, height, width)

    return sobel_magnitudes



def total_variation_loss(image):
    """Compute Total Variation Loss for a 3D image."""
    batch_size, n_views, channels, width, height = image.size()

    # Calculate differences between adjacent pixels in width (w) and height (h)
    tv_loss = torch.sum(torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])) + \
              torch.sum(torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1]))
    
    return tv_loss / (batch_size * n_views * channels * width * height)


def chamfer_distance(output, target):
    """
    Computes the Chamfer distance between output and target tensors.
    
    Args:
        output: Tensor of shape [batch, 4, 3, w, h], predicted coordinates.
        target: Tensor of shape [batch, 4, 3, w, h], ground truth coordinates. Background is marked by 0.

    Returns:
        Chamfer distance as a scalar value.
    """
    
    
    # Flatten the spatial dimensions for processing
    batch_size, viewpoints, coords, w, h = target.shape
    output = output.view(batch_size, viewpoints, coords, -1)  # [batch, 4, 3, w*h]
    target = target.view(batch_size, viewpoints, coords, -1)  # [batch, 4, 3, w*h]

    # Identify valid points in the target (non-background)
    valid_mask = (target != 0).any(dim=2)  # [batch, 4, w*h], True where target is valid
    valid_target = [target[b, v, :, valid_mask[b, v]] for b in range(batch_size) for v in range(viewpoints)]
    valid_output = [output[b, v, :, valid_mask[b, v]] for b in range(batch_size) for v in range(viewpoints)]
    
    chamfer_distances = []

    for t, o in zip(valid_target, valid_output):
        if t.shape[1] == 0 or o.shape[1] == 0:
            # No valid points to compare, skip this case
            chamfer_distances.append(0.0)
            continue
        
        # Transpose for easier point-wise comparisons
        t = t.T  # [N_t, 3]
        o = o.T  # [N_o, 3]

        # Compute distances from each point in target to nearest in output
        dist_t_to_o = torch.cdist(t.unsqueeze(0), o.unsqueeze(0), p=2).squeeze(0).min(dim=1)[0]
        dist_o_to_t = torch.cdist(o.unsqueeze(0), t.unsqueeze(0), p=2).squeeze(0).min(dim=1)[0]

        # Chamfer distance is the sum of both directions
        chamfer_distance = dist_t_to_o.mean() + dist_o_to_t.mean()
        chamfer_distances.append(chamfer_distance.item())

    return sum(chamfer_distances) / len(chamfer_distances)


import cv2
'''  Rewrite this to do delation operator for detected edges to enhance inner parts!!!

def erosion_operator(image,it):
    #kernel = np.asarray([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    kernel = np.ones([3,3],dtype=np.uint8)
    
    eroded_image = image.copy()
    for proj in range(0,4):
        for channel in range(0,3):
            eroded_image[proj][channel] = cv2.erode(eroded_image[proj][channel], kernel, iterations=it)
            
            eroded_part = image[proj][channel] - eroded_image[proj][channel]

            image[proj][channel] = np.where(eroded_part > 0.1, 0, image[proj][channel])

    
    return image
'''


def combined_loss(output, target, nn_loss, adverserial_loss, feature_matching_loss):
    loss = nn_loss(output, target)
    
    
    target_magnitude = compute_sobel_magnitude(target)
    #output_magnitude = compute_sobel_magnitude(output)
    #Sobel_magnitude_mse = torch.mean((output_magnitude - target_magnitude) ** 2)

    sobel_target_mask = (target_magnitude != 0)
    #sobel_target_mask = sobel_target_mask.expand_as(output)
    Sobel_magnitude_mse = torch.mean((output[sobel_target_mask] - target[sobel_target_mask]) ** 2)

    
    mask = (target != 0).all(dim=2)
    mask = mask.unsqueeze(2).expand_as(output)  # Shape: (batch, n_views, 3, w, h)

    object_loss = torch.mean((output[mask] - target[mask]) ** 2)

    #tv_loss = total_variation_loss(output)
    
    '''
    Vastagaítani a vonalakat!!! cv2 -> dilate
    '''
    #total_loss = 500 * Sobel_magnitude_mse + 5 * loss + 0 * object_loss + 40 * adverserial_loss
    total_loss = 200 * Sobel_magnitude_mse + 5 * loss + 0 * object_loss + 20 * adverserial_loss ---> This was 20250205_02_sobel_04 this is thebest so far
    #total_loss = 30 * Sobel_magnitude_mse + 2 * loss + 5 * object_loss + 3 * adverserial_loss ---> This was 20250205_02_sobel_02
    #total_loss = 3 * object_loss + 2 * loss + 10 * adverserial_loss   + 0 * tv_loss + 0*feature_matching_loss ---> This was loss for 20250204_02
    #total_loss = 5 * object_loss + 2 * loss + 5 * adverserial_loss   + 1 * tv_loss + 2*feature_matching_loss ---> This was loss for 20250204_01
    
    # full_target2: just 50*object loss+ 10*mse
    # full target3: total_loss = 50 * object_loss + 10 * loss + 25 * adverserial_loss   + 0.5 * tv_loss + 0*feature_matching_loss -> its a fail...
    # Next try of full_target: total_loss = 30 * object_loss + 10 * loss + 10 * adverserial_loss   + 5 * tv_loss + 15*feature_matching_loss feature matching did not help so set to 0
    # This is for fulltarget validationloss: loss = combined_loss(outputs, targets, nn_loss, 0,0)
    # fulltarget4 : total_loss = 5 * object_loss + 5 * loss + 20 * adverserial_loss   + 0 * tv_loss + 0*feature_matching_loss
    # Just adverserail
    return total_loss

def perceptual_loss(output, target, feature_extractor):
    losses = []
    for i in range(output.shape[1]):  # Loop through each view
        output_single_view = output[:, i, :, :, :].permute(0, 3, 1, 2) * 255.0  # Shape: [batch_size, 3, height, width]
        target_single_view = target[:, i, :, :, :].permute(0, 3, 1, 2) * 255.0  # Shape: [batch_size, 3, height, width]
        
        output_features = feature_extractor(output_single_view)  # [batch_size, features, height, width]
        target_features = feature_extractor(target_single_view)

        # Compute MSE for this view and add to losses list
        losses.append(F.mse_loss(output_features, target_features))

    # Average the perceptual losses over all views
    return torch.mean(torch.stack(losses))


 

def train_one_chunk2(model,optimizer, training_data_loader,test_data_loader,curr_chunk,number_of_chunks,device,scheduler=False):
    
    nn_loss = nn.MSELoss()

    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(training_data_loader):
        inputs,targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero gradients before backward pass

        
        outputs = model(inputs)
        
        #losses = [perceptual_loss(output, target, model.feature_extractor) for output, target in zip(outputs, targets)]
        loss = combined_loss(outputs,targets,nn_loss,perceptual_loss(outputs, targets, model.feature_extractor))
        #print(f'This is losses: {losses}')
        #loss = sum(losses) / len(losses) 
        
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # preventing exploding gradients...
        
        running_loss += loss.item()
    
    #scheduler.step() # This is for the schedular
    
    # Validation after each epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, targets in test_data_loader:
            inputs,targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)

            #losses = [perceptual_loss(output, target, model.feature_extractor) for output, target in zip(outputs, targets)]
            loss = combined_loss(outputs,targets,nn_loss,perceptual_loss(outputs, targets, model.feature_extractor))
            #loss = sum(losses) / len(losses)  # Average loss across views
            
            val_loss += loss.item()
    

    if scheduler:
        scheduler.step(val_loss)

    # Print average losses for this epoch
    
    
    return running_loss/len(training_data_loader),val_loss/len(test_data_loader)

def train_one_chunk_originalMVPCC(model, optimizer, training_data_loader, device,discriminator,discriminator_optimizer, scheduler=False):
    nn_loss = nn.SmoothL1Loss()
    criterion_for_discriminator = nn.MSELoss() # used to be BCE
    
    model.train()
    running_loss = 0.0
    running_loss_discriminator = 0.0
    for i, (inputs, targets) in enumerate(training_data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero gradients before backward pass

        inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
        targets = targets.permute(0, 1, 4, 2, 3).contiguous()
        
        outputs = model(inputs)
        
        
        #_,feature_matching_loss = train_discriminator(discriminator,device,2,targets,discriminator_optimizer,outputs,criterion_for_discriminator)
        adversarial_loss,f_m,loss_discriminator = train_discriminator(discriminator,device,2,targets,discriminator_optimizer,outputs,criterion_for_discriminator)
        
        outputs = outputs.reshape(inputs.shape[0], 4, 3, inputs.shape[3], inputs.shape[4])

        # Calculate combined loss
        loss = combined_loss(outputs, targets, nn_loss, adversarial_loss,f_m)
        
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Preventing exploding gradients...
        
        running_loss += loss.item()
        running_loss_discriminator += loss_discriminator

    return running_loss / len(training_data_loader) , running_loss_discriminator / len(training_data_loader)


from MVPCC_Networks import Discriminator
def train_discriminator(discriminator,device,batch_size,real_data,optimizer,Inpainted_inputs,criterion):
    real_data = real_data.reshape(real_data.shape[0], 4*3, real_data.shape[3], real_data.shape[4])

    real_label = torch.tensor(1.0, device=device)
    fake_label = torch.tensor(0.0, device=device)

    output_real,real_features = discriminator(real_data)
    real_label = torch.full_like(output_real,1.0,device=device)
    

    loss_real = criterion(output_real, real_label)  

    output_fake,fake_features = discriminator(Inpainted_inputs.detach())  
    fake_label = torch.full_like(output_fake,0.0,device=device)

    loss_fake = criterion(output_fake, fake_label)  

    #print(f'This is real_loss: {loss_real}')
    #print(f'This is loss_fake: {loss_fake}')

    feature_matching_loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        feature_matching_loss += torch.mean(torch.abs(real_feat - fake_feat))

    #print(f'This is feature matching loss: {feature_matching_loss}')
    # Combine discriminator losses
    loss_discriminator = (loss_real + loss_fake) 
    # I have changed it so fake loss has a bigger impact on the whole loss...
    
    # Backpropagation for the discriminator
    optimizer.zero_grad()
    loss_discriminator.backward()
    optimizer.step()

    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Preventing exploding gradients...

    return criterion(output_fake.detach(),real_label),feature_matching_loss.item(),loss_discriminator.item()

# Function for just training:
def train_one_chunk(model, optimizer, training_data_loader, device, scheduler=False):
    nn_loss = nn.MSELoss()

    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(training_data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero gradients before backward pass

        outputs = model(inputs)

        
        # Calculate combined loss
        loss = combined_loss(outputs, targets, nn_loss, perceptual_loss(outputs, targets, model.feature_extractor))

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Preventing exploding gradients...
        
        running_loss += loss.item()
    
    # Return average training loss
    return running_loss / len(training_data_loader)

# Function for validation/testing
def validate_one_chunk(model, test_data_loader, device, scheduler=False):
    nn_loss = nn.MSELoss()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.permute(0, 1, 4, 2, 3).contiguous()
            targets = targets.permute(0, 1, 4, 2, 3).contiguous()
            
            outputs = model(inputs)
            outputs = outputs.reshape(inputs.shape[0], 4, 3, inputs.shape[3], inputs.shape[4])
            loss = combined_loss(outputs, targets, nn_loss, 0,0)
            # Calculate combined loss


            val_loss += loss.item()
    
    # Return average validation loss
    return val_loss 

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


def create_dataloader(dataset,batch_size = 32, train_ratio=1):

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader


from predict import predict
import numpy as np

def train(dataset_size,number_of_incomplete_samples,number_of_chunks = 60,learning_rate=0.002,epochs = 50,batch_size = 4,schedular=False):
    name = "20250929_new"
    
    chunk_size = dataset_size // number_of_chunks
    
    test_output = np.load("XYZ_projections_sparse\XYZ_66.npy")
    test_output = test_output.transpose(0, 3, 1, 2).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')
    model = InpaintGenerator(residual_blocks=8).to(device)
    if schedular:
        optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.95)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = False

    # This part is for discriminator:
    discriminator = Discriminator(3*4).to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # ------------------------------

    train_losses = []
    test_losses = []

    for epoch in range(1,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        for i in range(0, number_of_chunks):

            #dataset = create_dataset(i*chunk_size,(i+1)*chunk_size,number_of_incomplete_samples,incomplete_dir="sparse_depth_maps")
            
            #train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)

            train_dataloader, test_dataloader = load_dataloader(i)

            train_loss,discriminator_loss = train_one_chunk_originalMVPCC(model,optimizer,train_dataloader,device,discriminator,discriminator_optimizer,scheduler=schedular)
            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {train_loss:.4f} - Discriminator Loss: {discriminator_loss:.4f}")

            train_losses.append(train_loss)
            test_loss = validate_one_chunk(model,test_dataloader,device,scheduler)
            test_losses.append(test_loss)

            if epoch == 1 and i == 0:
                model.eval()  # Set to evaluation mode
                with torch.no_grad():  # Disable gradient computation for prediction
                    out = predict(model, device, test_output)
                out = predict(model,device,test_output)
                out = out.cpu().detach().numpy().squeeze()  # Convert to NumPy array
                out = out.reshape(4, 3, 256, 256)[0]
                save_image(out,"test_images\\"+name+"\\first_chunk_epoch1_mvpcc.png")
                model.train()
            
        print(f"Epoch [{epoch}/{epochs}] - Test Loss: {test_loss:.4f}")
        
        
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = predict(model,device,test_output)
            out = out.cpu().detach().numpy().squeeze()  # Convert to NumPy array
            out = out.reshape(4, 3, 256,256)[0]
            save_image(out,"test_images\\"+name+"\\"+str(epoch)+".png")
            model.train()

        # Create one projection output:
         
    torch.save(model.state_dict(), "trained_model/"+name+".pth")

    train_losses = np.asarray(train_losses).reshape(number_of_chunks, epoch)
    test_losses = np.asarray(test_losses).reshape(number_of_chunks, epoch)

    # Compute the mean along the first axis (averaging over the 32 chunks)
    train_losses_avg = train_losses.mean(axis=0)
    test_losses_avg = test_losses.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_avg, label='Train Loss')
    plt.plot(test_losses_avg, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    
    # Save the plot
    plt.savefig(name+".png")
    plt.close()

def manage_dataloaders(dataset_size,number_of_incomplete_samples,number_of_chunks = 60,batch_size = 4):
    chunk_size = dataset_size // number_of_chunks
    

    for i in tqdm(range(number_of_chunks)):

        dataset = create_dataset2(i*chunk_size,(i+1)*chunk_size)
        
        train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)
        save_dataloader(train_dataloader,i,"train")
        save_dataloader(test_dataloader,i,"test")


batch=2
manage_dataloaders(128,1,16,batch)
train(128,1,16,batch_size=batch,epochs=200)

# 16 chunks: 1 chunk 16 ships?