import torch
from sklearn.metrics import roc_auc_score, average_precision_score
def validate_model(model, val_loader,device):
    model.eval()
    preds = []
    labels = []
    total_loss = 0
    vdist = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for i, data in enumerate(val_loader):

        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            out = model(x)
            loss = criterion(out,y)

        total_loss = total_loss + loss.item()
        
        probs = torch.sigmoid(out)
        
        preds.extend(probs.cpu().numpy())
        labels.extend(y.cpu().numpy())

        del x, y, loss
        torch.cuda.empty_cache()
        
    loss = total_loss/len(val_loader)
    auroc = roc_auc_score(labels,preds)
    aupr = average_precision_score(labels,preds)

    return loss,auroc,aupr