import os
import argparse
import random
import math # 
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timeit import default_timer
from sklearn import metrics
from vq import Encoder_v2, Encoder_v3
# from task_configs import get_data
# from utils import calculate_auroc, calculate_aupr, auroc, inverse_score
# from utils import auroc_aupr,inverse_two_scores #
from wrn1d import ResNet1D, ResNet1D_v2 # 
import copy

# torch.cuda.set_device(0)
print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", DEVICE)



#-----------------------Configurations
configs = {'weight':'unet', # resnet, unet, nas-deepsea ,
           'one_hot':False,
           'lr':0.0005,
           'optimizer': 'Adam',
           'weight_decay': 0.00001,
           'batch_size':32,
           'epochs':100,
            'channels': [16,32,64],
            'drop_out':0.2}
print(configs)
# weight: nas-deepsea, one_hot True, lr 0,01

root = "/home/wenduoc/ORCA/clean/gene-orca/datasets"

#--------------------------Metric
def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    score = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, score
def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr

def auroc_aupr(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()

    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc, avg_aupr

class inverse_two_scores(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, output, target):
        return 1 - self.score_func(output, target)[0], 1 - self.score_func(output, target)[1]

#-----------------------Data loaders
def load_deepsea(root, batch_size, one_hot = True, valid_split=-1,rc_aug=False, shift_aug=False):
    filename = root + '/deepsea_filtered.npz'

    if not os.path.isfile(filename):
        with open(filename, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz").content)

    data = np.load(filename)

    y_train = torch.from_numpy(np.concatenate((data['y_train'], data['y_val']), axis=0)).float() 
    y_test = torch.from_numpy(data['y_test']).float()   # shape = (149400, 36)
    if one_hot:
        x_train = torch.from_numpy(np.concatenate((data['x_train'], data['x_val']), axis=0)).transpose(-1, -2).float()  
        x_test = torch.from_numpy(data['x_test']).transpose(-1, -2).float()  # shape = (149400, 1000, 4)
    else:
        x_train = torch.from_numpy(np.argmax(np.concatenate((data['x_train'], data['x_val']), axis=0), axis=2)).unsqueeze(-2).float()
        x_test = torch.from_numpy(np.argmax(data['x_test'], axis=2)).unsqueeze(-2).float()

        if rc_aug:
            print('reverse complement')
            x_train2 = copy.deepcopy(x_train) 
            y_train2 = copy.deepcopy(y_train)

            x_train = torch.concatenate(x_train,x_train2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return train_loader, None, test_loader

def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False, quantize=False,rc_aug=False,shift_aug=False, one_hot=True):
    data_kwargs = None

    if dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size,one_hot = one_hot, valid_split=valid_split,rc_aug=rc_aug, shift_aug=shift_aug)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs

#-----------------------Model
class DeepSEA(nn.Module):
    def __init__(self, ):
        super(DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding = 4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding = 4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding = 4)
        self.Conv4 = nn.Conv1d(in_channels=960, out_channels=768, kernel_size=8, padding = 4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(64*768, 36)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.Conv4(x)
        x = self.flatten(x)
        x = self.Linear(x)
  
        return x


class DeepSEA2(nn.Module):
    def __init__(self, ):
        super(DeepSEA2, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding = 4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding = 4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding = 4)
        self.Conv4 = nn.Conv1d(in_channels=960, out_channels=768, kernel_size=8, padding = 4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(64*768, 768)
        self.Linear2 = nn.Linear(768, 919)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.Conv4(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
  
        return x

class DeepSEA_Original(nn.Module):
    def __init__(self, ):
        super(DeepSEA_Original, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(53*960, 925) 
        self.Linear2 = nn.Linear(925, 919)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
  
        return x

class DeepSEA_Full(nn.Module):
    def __init__(self, ):
        super(DeepSEA_Full, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding=4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding=4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding=4)
        self.Conv4 = nn.Conv1d(in_channels=960, out_channels=768, kernel_size=8, padding=4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(64*768, 925)
        self.Linear2 = nn.Linear(925, 919)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.Conv4(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
  
        return x

# NAS-BENCH-360 DeepSea
class NAS_DeepSEA(nn.Module):
    def __init__(self, ):
        super(NAS_DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding = 4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding = 4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding = 4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(63*960, 925) 
        self.Linear2 = nn.Linear(925, 36)
        

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
  
        return x

class Baseline(nn.Module):
    def __init__(self, ):
        super(Baseline, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=768, kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(768*500, 36)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = self.flatten(x)
        x = self.Linear(x)
        return x

class DASH_DEEPSEA(nn.Module):
    def __init__(self, ks=None,ds=None):
        super(DASH_DEEPSEA, self).__init__()
        k, d = 8 if ks is None else ks[0], 1 if ds is None else ds[0]
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[1], 1 if ds is None else ds[1]
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[2], 1 if ds is None else ds[2]
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=k, dilation=d, padding=k//2 * d)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(59520, 925)
        self.linear2 = nn.Linear(925, 36)

    def forward(self, input):
        # print("input", input.shape)
        s = input.shape[-1]
        x = self.conv1(input)[..., :s]
        x = F.relu(x)
        # print("1", x.shape)
        x = self.maxpool(x)
        # print("2", x.shape)
        x = self.drop1(x)
        # print("3", x.shape)
        s = x.shape[-1]
        x = self.conv2(x)[..., :s]
        # print("4", x.shape)
        x = F.relu(x)
        x = self.maxpool(x)
        # print("5", x.shape)
        x = self.drop1(x)
        # print("6", x.shape)
        s = x.shape[-1]
        x = self.conv3(x)[..., :s]
        # print("7", x.shape)
        x = F.relu(x)
        x = self.drop2(x)
        # print("8", x.shape)
        x = x.view(x.size(0), -1)
        # print("9", x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x




if configs['weight']=='nas-deepsea':
    model = NAS_DeepSEA().to(DEVICE)
elif configs['weight']=='unet':
    if not configs['one_hot']:
        model = Encoder_v3(768, channels = configs['channels'], dropout=configs['drop_out'], f_channel=1000,num_class=36,ks=None,ds=None,downsample=False,seqlen=1000)
    else: 
        model = Encoder_v2(4, channels = configs['channels'], dropout=configs['drop_out'], f_channel=1000,num_class=36,ks=None,ds=None,downsample=False,seqlen=1000)
    model = model.to(DEVICE)
elif configs['weight']=='resnet':
    in_channel=4
    num_classes=36
    # num_classes=919
    mid_channels=min(4 ** (num_classes // 10 + 1), 64)
    dropout=0
    ks=[15, 19, 19, 7, 7, 7, 19, 19, 19]
    ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]
    # ks=[19, 19, 19, 19, 19, 19, 11, 19, 19]
    # ds=[1, 1, 15, 1, 15, 7, 15, 15, 1]
    activation=None
    remain_shape=False
    model = ResNet1D_v2(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=configs['drop_out'], ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, input_shape=(configs['batch_size'],4,1000), embed_dim=768).to(DEVICE)
    # model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape).to(DEVICE)
print(model)

if configs['optimizer']=='SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=configs["lr"], momentum=0.9, weight_decay=0) # momentum 0.9
elif configs['optimizer']=='Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), weight_decay=configs['weight_decay'])

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=0)
base, accum = 0.2, 1
# sched = [30, 60, 90, 120, 160]
sched = [20, 40, 60]
def weight_sched_train(epoch):    
    optim_factor = 0
    for i in range(len(sched)):
        if epoch > sched[len(sched) - 1 - i]:
            optim_factor = len(sched) - i
            break
    return math.pow(base, optim_factor)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = weight_sched_train)

# loss
loss = torch.nn.BCEWithLogitsLoss(pos_weight = 4 * torch.ones((36, )).to(DEVICE))
# loss = torch.nn.BCEWithLogitsLoss().to(DEVICE)
# loss = torch.nn.CrossEntropyLoss().to(DEVICE)



# get data
train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA', batch_size=configs['batch_size'],valid_split=False, one_hot=configs['one_hot'])
# train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA_FULL', 256, False)
# train_loader, val_loader, test_loader = load_deepsea(root, configs['batch_size'], one_hot = True, valid_split=-1,rc_aug=False, shift_aug=False)

for batch in train_loader: 
        x, y = batch
        print('x:',x.size())
        print('y:',y.size())
        break

# adapt from main.py
def evaluate(model, loader, loss, metric, n_eval, two_metrics=False):
    model.eval()
    eval_batch_size = 1000
    eval_loss, eval_score = 0, 0
    eval_score2 = 0
    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            outs.append(out)
            ys.append(y)

            n_data += x.shape[0]
  
            if two_metrics:
                if n_data >= eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys)[0].item()
                    eval_score2 += metric(outs, ys)[1].item()
                    n_eval += 1

                    ys, outs, n_data = [], [], 0
            else:
                if n_data >= eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    # print('429 outs',outs)
                    # print('429 ys',ys)
                    eval_score += metric(outs, ys)[0].item()
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

        eval_loss /= n_eval
        eval_score /= n_eval
        eval_score2 /= n_eval
    if two_metrics:
        return eval_loss, eval_score, eval_score2
    else:
        return eval_loss, eval_score

def train_one_epoch(model, optimizer, scheduler, loader, loss, temp):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        x, y = data 
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        l = loss(out, y)
        l.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
       
        optimizer.step()
        optimizer.zero_grad()
        
        # scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break

    # scheduler.step()

    return train_loss / temp


print("\n------- Start Training --------")
# training and validating


metric = inverse_two_scores(auroc_aupr)

train_time, train_score, train_losses = [], [], []

for ep in range(configs['epochs']):
    # train
    time_start = default_timer()
    train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss, n_train)
    train_time_ep = default_timer() -  time_start 
    # val    
    val_loss, val_score = evaluate(model, val_loader, loss, metric, n_val)
    
    train_losses.append(train_loss)
    train_score.append(val_score)
    train_time.append(train_time_ep)

    scheduler.step()
    
    print("[train", "full", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % np.min(train_score))
    
    if np.min(train_score) == val_score:
        torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_score': val_score, 
                  'epoch': ep}, os.path.join('/home/wenduoc/automation/automation/deepsea/', 'pretrained_model.pth'))

np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_losses.npy'), train_losses)
np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_score.npy'), train_score)    
np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_time.npy'), train_time) 

# test
print("\n------- Start Test --------")
test_scores = []
test_scores2 = []

test_model = model
test_time_start = default_timer()
test_loss, test_score, test_score2 = evaluate(test_model, test_loader, loss, metric, n_test, two_metrics=True)
test_time_end = default_timer()
test_scores.append(test_score)
test_scores2.append(test_score2)
print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score, "\tsecond test score:", "%.4f" % test_score2)




