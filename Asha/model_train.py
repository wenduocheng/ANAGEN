import torch
def train_model(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for i, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad() 
        
        x, y = x.to(device), y.to(device)
        
    
        out = model(x)
        loss = criterion(out,y)

        total_loss = total_loss + loss.item()
        loss.backward() 
        optimizer.step() 
        
        total_loss += loss.item()  
        
        del x, y, loss
        torch.cuda.empty_cache()

    return total_loss / len(train_loader.dataset)
