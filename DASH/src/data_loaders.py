import torch
import mat73
import scipy.io

def load_deepsea(batch_size):
    train_data = mat73.loadmat('/home/ubuntu/automation/DASH/src/data/train.mat')
    valid_data = scipy.io.loadmat('/home/ubuntu/automation/DASH/src/data/valid.mat')
    test_data = scipy.io.loadmat('/home/ubuntu/automation/DASH/src/data/test.mat')
    

    x_train = torch.FloatTensor(train_data['trainxdata']) # 
    y_train= torch.FloatTensor(train_data['traindata']) #
    x_val = torch.FloatTensor(valid_data['validxdata']) # 
    y_val= torch.FloatTensor(valid_data['validdata']) # 
    x_test = torch.FloatTensor(test_data['testxdata']) # 
    y_test= torch.FloatTensor(test_data['testdata']) # 

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Train dataset samples = {}, batches = {}".format(x_train.shape[0], len(train_loader)))
    print("Valid dataset samples = {}, batches = {}".format(x_val.shape[0], len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(x_test.shape[0], len(test_loader)))

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_deepsea(batch_size=32)
    # train_loader, val_loader, test_loader = load_deepsea(batch_size=32)