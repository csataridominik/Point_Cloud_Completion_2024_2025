from model import MultiViewCompletionNet
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import create_dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle



def total_variation_loss(image):

    """Compute Total Variation Loss for a 2D image."""
    batch_size, _, height, width = image.size()

    # Calculate differences between adjacent pixels
    tv_loss = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])) + \
              torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    
    return tv_loss / (batch_size * height * width)

def combined_loss(output, target,nn_loss,preceptual_loss,lambda_tv=0.3):

    loss = nn_loss(output, target)

    # This part is to check loss only where there is an object...
    mask = (target >= 0)
    object_loss = torch.mean((output[mask] - target[mask]) ** 2)
    
    tv_loss = total_variation_loss(output)

    total_loss = 50 * object_loss + 5 * loss + 1.5 * preceptual_loss # not using tv_loss in this implemantation part....
    return total_loss


def perceptual_loss(output, target, feature_extractor):
    losses = []
    for i in range(output.shape[1]):  # Loop through each view
        output_single_view = output[:, i, :, :].unsqueeze(1)* 255.0  # Shape: [batch_size, 1, height, width]
        target_single_view = target[:, i, :, :].unsqueeze(1)* 255.0  # Shape: [batch_size, 1, height, width]
        
        # Repeat channels to simulate RGB input
        output_features = feature_extractor(output_single_view.repeat(1, 3, 1, 1))  # [batch_size, 3, height, width]
        target_features = feature_extractor(target_single_view.repeat(1, 3, 1, 1))


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

import torch
import torch.nn as nn

# Function for training
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
            
            outputs = model(inputs)

            # Calculate combined loss
            loss = combined_loss(outputs, targets, nn_loss, perceptual_loss(outputs, targets, model.feature_extractor))

            val_loss += loss.item()
    
    # Return average validation loss
    return val_loss / len(test_data_loader)




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


def create_dataloader(dataset,batch_size = 32, train_ratio=0.8):

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader

def train(dataset_size,number_of_incomplete_samples,number_of_chunks = 60,learning_rate=0.001,epochs = 50,batch_size = 4,schedular=False):
    chunk_size = dataset_size // number_of_chunks
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')
    model = MultiViewCompletionNet().to(device)
    if schedular:
        optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.95)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = False

    train_losses = []
    test_losses = []

    for epoch in range(1,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        for i in range(0, number_of_chunks):

            #dataset = create_dataset(i*chunk_size,(i+1)*chunk_size,number_of_incomplete_samples,incomplete_dir="sparse_depth_maps")
            
            #train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)

            train_dataloader, test_dataloader = load_dataloader(i)

            train_loss = train_one_chunk(model,optimizer,train_dataloader,device,scheduler)
            print(f"Chunk [{i+1}/{number_of_chunks}] - Training Loss: {train_loss:.4f}")

            train_losses.append(train_loss)
            

        test_loss = validate_one_chunk(model,test_dataloader,device,scheduler)
        print(f"Chunk [{epoch}/{epochs}] - Test Loss: {test_loss:.4f}")
        test_losses.append(test_loss)

        
    torch.save(model.state_dict(), "trained_model/trained_model_object_loss.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    #plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Chunks Processed')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss over Epochs')
    
    # Save the plot
    plt.savefig("trained_model_object_loss.png")
    plt.close()

def manage_dataloaders(dataset_size,number_of_incomplete_samples,number_of_chunks = 60,batch_size = 4):
    chunk_size = dataset_size // number_of_chunks
    

    for i in tqdm(range(number_of_chunks)):

        dataset = create_dataset(i*chunk_size,(i+1)*chunk_size,number_of_incomplete_samples,incomplete_dir="sparse_depth_maps")
        
        train_dataloader, test_dataloader = create_dataloader(dataset,batch_size)
        save_dataloader(train_dataloader,i,"train")
        save_dataloader(test_dataloader,i,"test")



#manage_dataloaders(256,1,16,4)
#train(256,1,16,batch_size=4,epochs=20)