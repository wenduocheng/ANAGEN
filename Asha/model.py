import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from task_configs import get_config, get_model
from config import configs

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
    
    
    
def get_model_from_nas():
    print(f'Loading model from NAS result.')
    # nas_result = np.load("/home/ec2-user/automation/final_res.npz")
    nas_result = np.load(configs()['dash_res_path'])
    ks = nas_result["kernel_choices"]
    ds = nas_result["dilation_choices"]
    dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(dataset="deepsea")
    model = get_model(arch='wrn', sample_shape=sample_shape, num_classes=num_classes, config_kwargs=config_kwargs, ks=ks, ds=ds)
    return model

 
def get_model_result(model = 'nas'):
    if model == 'deepsea':
        return DeepSEA_Original()
    else:
        model = get_model_from_nas()
        return model