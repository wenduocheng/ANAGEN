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
        feats = torch.from_numpy(feats).float()
        label = torch.from_numpy(label).float()

        return feats ,label

    def __len__(self):
        return len(self.X)

def load_deepsea(batch_size):
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

if __name__ == "__main__":
    train_loader, valid_loader, test_loader = load_deepsea(batch_size=32)