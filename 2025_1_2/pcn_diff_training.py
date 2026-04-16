from training import manage_dataloaders,load_dataloader,update_loss_png,sample_farthest_points

import torch
from time import time
from models_fun import DiffusionModel

def validation(model,device,validation_dataloader):
    val_loss = 0
    iter = 0
    model.eval()
    with torch.no_grad():
        for x,y in validation_dataloader:
            x,_ = sample_farthest_points(x, K=4*1024, random_start_point=False)
            y,_ = sample_farthest_points(x, K=4*1024, random_start_point=False)

            # Move to device
            x = x.to(device).transpose(1, 2)   # [B, 3, 4096]
            y = y.to(device).transpose(1, 2)   # [B, 3, 16384] (ground truth full cloud)

            
            _,m = prepare_inputs(x,y)

            ts = torch.randint(0, model.time_steps, (x.shape[0],), device=device)
            y_noised, noise = model.add_noise(y, ts)     
            xy = torch.cat([x,y_noised],dim=2)
            # model predicts noise

            xy = torch.cat([xy, m], dim=1)
            pred_noise = model(xy, ts)

            loss = ((pred_noise - noise)**2).mean()

            val_loss += loss.item()
            iter +=1

    return val_loss/iter

def prepare_inputs(x, y):
    B, C, N = x.shape
    xy = torch.cat([x, y], dim=2)

    mask = torch.cat([torch.ones((B, 1, N), device=x.device),
                      torch.zeros((B, 1, y.shape[2]), device=x.device)], dim=2)
    
    return xy, mask

def train(name,dataset_size,number_of_chunks = 60,learning_rate=0.001,batch_size=4,epochs = 50,load_checkpoint="",val_chunks = 1):
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')

    model = DiffusionModel()

    opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    torch.backends.cudnn.benchmark = True
    model.to(device)

    losses = []            
    val_losses = []
    best_loss = 1e8
    if load_checkpoint != "":
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
        val_losses = checkpoint['val_loss']
    else:
        start_epoch = 1

    for epoch in range(start_epoch,epochs+1):
        print("-----------------------------------------------------------")
        print(f'This is Epoch: {epoch}/{epochs}...')
        model.train()
        stime = time()
        chunk_loss = 0
        temp_val_losses = 0
        for i in range(0, number_of_chunks):
            train_dataloader = load_dataloader(i)

            if i >= number_of_chunks-val_chunks: 
                temp_val_losses += validation(model,device,train_dataloader)
                
                continue

            loader_loss = 0
            total = 0

            for it_, (x, y) in enumerate(train_dataloader):
                x,_ = sample_farthest_points(x, K=4*1024, random_start_point=False)
                y,_ = sample_farthest_points(x, K=4*1024, random_start_point=False)

                # Move to device
                x = x.to(device).transpose(1, 2)   # [B, 3, 4096]
                y = y.to(device).transpose(1, 2)   # [B, 3, 16384] (ground truth full cloud)

                opt.zero_grad()

                _,m = prepare_inputs(x,y)

                ts = torch.randint(0, model.time_steps, (x.shape[0],), device=device)
                y_noised, noise = model.add_noise(y, ts)     
                xy = torch.cat([x,y_noised],dim=2)
                # model predicts noise

                xy = torch.cat([xy, m], dim=1)
                pred_noise = model(xy, ts)

                loss = ((pred_noise - noise)**2).mean()

                loss.backward()
                opt.step()

                loader_loss += loss.item()
                total += 1

                #scheduler.step() # THIS IS FOR CYVLEIC!!!!!!!!!!!!!!!!!!!
            
            chunk_loss += loader_loss/total
            
        losses.append(chunk_loss/number_of_chunks)
        val_losses.append(temp_val_losses/val_chunks)

        #scheduler.step(temp_val_losses)
        temp_val_losses = 0

        ftime = time()
        print(f"Epoch [{epoch}/{epochs}] trained in {ftime - stime}s; Training loss => {losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")
        #print(f'learning_rate: {scheduler.get_last_lr()}')

        if epoch % 5 == 0 or epoch == 1:
            update_loss_png(losses, val_losses,epoch, name)
            
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
        
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'val_loss': val_losses
            }
            torch.save(checkpoint, "checkpoints/" + name + ".pth")
    
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'losses': losses,
    'val_loss': val_losses
    }
    torch.save(checkpoint, "checkpoints/" + name + "_final.pth")



def  setup_and_train(name,setup=True, checkpoint="",):
    batch = 16
    dataset_size =  1700*5
    number_of_chunks = 16 
    val_chunks=2
    if setup:
        manage_dataloaders(dataset_size,number_of_chunks,batch)
    train(name,dataset_size,number_of_chunks,batch_size=batch,epochs=1_000,learning_rate=1e-4,load_checkpoint=checkpoint,val_chunks=val_chunks)


if __name__ == "__main__":
    setup_and_train("Diff_model_base",setup=False,checkpoint="")