import torch
import h5py
import scipy.io
import numpy as np

class DeepseaDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        feats = self.X[idx]
        label = self.y[idx]
        feats = torch.from_numpy(feats).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)

        return feats ,label

    def __len__(self):
        return len(self.X)

def load_deepsea_original(batch_size):
    path = "/home/ubuntu/automation/DASH/src/data/"

    with h5py.File(path+"train.mat", 'r') as file:
        x_train = file['trainxdata']
        y_train = file['traindata']
        x_train = np.transpose(x_train, (2, 1, 0))    
        y_train = np.transpose(y_train, (1, 0))   

    valid_data = scipy.io.loadmat(path+"valid.mat")
    test_data = scipy.io.loadmat(path+"test.mat")
    x_valid = valid_data["validxdata"]
    y_valid = valid_data["validdata"]
    x_test = test_data["testxdata"]
    y_test = test_data["testdata"]

    train_dataset = DeepseaDataset(x_train, y_train)
    valid_dataset = DeepseaDataset(x_valid, y_valid)
    test_dataset = DeepseaDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("No. of train samples = {}, batches = {}".format(train_dataset.__len__(), train_loader.__len__()))
    print("No. of valid samples = {}, batches = {}".format(valid_dataset.__len__(), valid_loader.__len__()))
    print("No. of test samples = {}, batches = {}".format(test_dataset.__len__(), test_loader.__len__()))

    return train_loader, valid_loader, test_loader

def load_deepsea(batch_size, one_hot = True, valid_split=1):
    filename = '/home/ubuntu/automation/DASH/src/data/deepsea_filtered.npz'
    data = np.load(filename)

    if valid_split > 0:
        if one_hot:
            x_train = torch.from_numpy(data['x_train']).transpose(-1, -2).float()  
        else:
            x_train = torch.from_numpy(np.argmax(data['x_train'], axis=2)).unsqueeze(-2).float()
        y_train = torch.from_numpy(data['y_train']).float() 
        if one_hot:
            x_val = torch.from_numpy(data['x_val']).transpose(-1, -2).float()   # shape = (2490, 1000, 4)
        else:
            x_val = torch.from_numpy(np.argmax(data['x_val'], axis=2)).unsqueeze(-2).float() 
        y_val = torch.from_numpy(data['y_val']).float()  # shape = (2490, 36)

    else:
        if one_hot:
            x_train = torch.from_numpy(np.concatenate((data['x_train'], data['x_val']), axis=0)).transpose(-1, -2).float()  
        else:
            x_train = torch.from_numpy(np.argmax(np.concatenate((data['x_train'], data['x_val']), axis=0), axis=2)).unsqueeze(-2).float()
        y_train = torch.from_numpy(np.concatenate((data['y_train'], data['y_val']), axis=0)).float() 

    if one_hot:
        x_test = torch.from_numpy(data['x_test']).transpose(-1, -2).float()  # shape = (149400, 1000, 4)
    else:
        x_test = torch.from_numpy(np.argmax(data['x_test'], axis=2)).unsqueeze(-2).float()
    y_test = torch.from_numpy(data['y_test']).float()   # shape = (149400, 36)
    
    if valid_split > 0:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, None, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = load_deepsea(batch_size=32)