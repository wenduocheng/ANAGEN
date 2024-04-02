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
from task_configs import get_data
from utils import calculate_auroc, calculate_aupr, auroc, inverse_score
from utils import auroc_aupr,inverse_two_scores #
from wrn1d import ResNet1D, ResNet1D_v2 # 

torch.cuda.set_device(3)
print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", DEVICE)

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

# model = DeepSEA().to(DEVICE) 
# model = DeepSEA_Full().to(DEVICE)
# model = NAS_DeepSEA().to(DEVICE)
# model = Baseline().to(DEVICE)
# model = DASH_DEEPSEA(ks=[15, 19, 19, 7, 7, 7, 19, 19, 19],ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]).to(DEVICE)  
# in_channel=4
# # num_classes=36
# num_classes=919
# mid_channels=min(4 ** (num_classes // 10 + 1), 64)
# dropout=0
# ks=[15, 19, 19, 7, 7, 7, 19, 19, 19]
# ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]
# # ks=[19, 19, 19, 19, 19, 19, 11, 19, 19]
# # ds=[1, 1, 15, 1, 15, 7, 15, 15, 1]
# activation=None
# remain_shape=False
# model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape).to(DEVICE)
# print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=0) # momentum 0.9

# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=0)
# base, accum = 0.2, 1
# sched = [30, 60, 90, 120, 160]
# def weight_sched_train(epoch):    
#     optim_factor = 0
#     for i in range(len(sched)):
#         if epoch > sched[len(sched) - 1 - i]:
#             optim_factor = len(sched) - i
#             break
#     return math.pow(base, optim_factor)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = weight_sched_train)

# loss
loss = torch.nn.BCEWithLogitsLoss(pos_weight = 4 * torch.ones((36, )).to(DEVICE))
# loss = torch.nn.BCEWithLogitsLoss(pos_weight = 4 * torch.ones((919, )).to(DEVICE))
# loss = torch.nn.BCEWithLogitsLoss()

EPOCH=200


root = '/home/wenduoc/ORCA/src/datasets'
# path = '/home/wenduoc/ORCA/src/pretrained_embedders/deepsea_DASH_full/2/' 
# if not os.path.exists(path):
#     os.makedirs(path)

# get data
train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA', 256, False)
# train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA_FULL', 256, False)

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
# adapt from main.py
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
       
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

# train_time, train_score, train_losses = [], [], []

# for ep in range(EPOCH):
#     # train
#     time_start = default_timer()
#     train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss, n_train)
#     train_time_ep = default_timer() -  time_start 
#     # val    
#     val_loss, val_score = evaluate(model, val_loader, loss, metric, n_val)
    
#     train_losses.append(train_loss)
#     train_score.append(val_score)
#     train_time.append(train_time_ep)

#     scheduler.step(val_loss)
    
#     print("[train", "full", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % np.min(train_score))
    
#     if np.min(train_score) == val_score:
#         torch.save({'model_state_dict':model.state_dict(),
#                   'optimizer_state_dict':optimizer.state_dict(),
#                   'scheduler_state_dict':scheduler.state_dict(),
#                   'val_score': val_score, 
#                   'epoch': ep}, os.path.join(path, 'pretrained_model.pth'))

# np.save(os.path.join(path, 'train_losses.npy'), train_losses)
# np.save(os.path.join(path, 'train_score.npy'), train_score)    
# np.save(os.path.join(path, 'train_time.npy'), train_time) 

# test
print("\n------- Start Test --------")
test_scores = []
test_scores2 = []

# test_model = model
# test_time_start = default_timer()
# test_loss, test_score, test_score2 = evaluate(test_model, test_loader, loss, metric, n_test, two_metrics=True)
# test_time_end = default_timer()
# test_scores.append(test_score)
# test_scores2.append(test_score2)
# print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score, "\tsecond test score:", "%.4f" % test_score2)


test_model = DeepSEA().to(DEVICE)  
# test_model = DeepSEA_Full().to(DEVICE) 
# test_model  = NAS_DeepSEA().to(DEVICE)
# test_model = Baseline().to(DEVICE) 
# test_model = DeepSEA_Original().to(DEVICE)
# test_model = DASH_DEEPSEA(ks=[15, 19, 19, 7, 7, 7, 19, 19, 19],ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]).to(DEVICE)  
# test_model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape).to(DEVICE)

# checkpoint = torch.load(os.path.join(path, 'pretrained_model.pth'))
# test_model.load_state_dict(checkpoint['model_state_dict'])

# trained = torch.load("/home/wenduoc/ORCA/src_ablations/pretrained_embedders/deepsea_full_ablations/405/best_deepsea_full.pth")
trained = torch.load("/home/wenduoc/ORCA/src/pretrained_embedders/deepsea/0/pretrained_model.pth")
test_model.load_state_dict(trained['model_state_dict'])

# load Wendy's trained DeepSEA Original Model
# trained_deepsea = torch.load(os.path.join(path, 'best_deepsea_full.pth'))
# test_model.Conv1.weight.data.copy_(trained_deepsea['0.weight'].squeeze(2))
# test_model.Conv1.bias.data.copy_(trained_deepsea['0.bias'])
# test_model.Conv2.weight.data.copy_(trained_deepsea['3.weight'].squeeze(2))
# test_model.Conv2.bias.data.copy_(trained_deepsea['3.bias'])
# test_model.Conv3.weight.data.copy_(trained_deepsea['6.weight'].squeeze(2))
# test_model.Conv3.bias.data.copy_(trained_deepsea['6.bias'])
# test_model.Linear1.weight.data.copy_(trained_deepsea['9.1.weight'])
# test_model.Linear1.bias.data.copy_(trained_deepsea['9.1.bias'])
# test_model.Linear2.weight.data.copy_(trained_deepsea['10.1.weight'])
# test_model.Linear2.bias.data.copy_(trained_deepsea['10.1.bias'])

test_time_start = default_timer()
test_loss, test_score, test_score2 = evaluate(test_model, test_loader, loss, metric, n_test, two_metrics=True)
test_time_end = default_timer()
test_scores.append(test_score)
test_scores2.append(test_score2)
print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score, "\tsecond test score:", "%.4f" % test_score2)

# np.save(os.path.join(path, 'test_scores.npy'), test_scores)
# np.save(os.path.join(path, 'test_score2.npy'), test_scores2)

