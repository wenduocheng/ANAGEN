import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from itertools import product
from functools import reduce, partial

from networks.deepsea import DeepSEA
from networks.wrn1d import ResNet1D



def get_model(arch, sample_shape, num_classes, config_kwargs, ks = None, ds = None, dropout = None):
    in_channel = sample_shape[1]
    activation = config_kwargs['activation']
    remain_shape = config_kwargs['remain_shape']
    if dropout is None:
        dropout = config_kwargs['dropout']
    pool_k = config_kwargs['pool_k']
    squeeze = config_kwargs['squeeze']

    mid_channels = min(4 ** (num_classes // 10 + 1), 64)

    # 1D model arch
    if arch == 'deepsea':
        model = DeepSEA(ks = ks, ds = ds)
    elif arch == 'wrn':
        model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
   
    return model


def get_config(dataset):
    einsum = True
    base, accum = 0.2, 1
    validation_freq = 1
    clip, retrain_clip = 1, -1
    quick_search, quick_retrain = 0.2, 1
    config_kwargs = {'temp': 1, 'arch_retrain_default': None, 'grad_scale': 100, 'activation': None, 'remain_shape': False, 'pool_k': 8, 'squeeze': False, 'dropout': 0}
    
    if dataset == "deepsea":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19, 23, 27, 31], [1, 2, 3, 5, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

        batch_size = 64
        arch_default = 'wrn'  
        config_kwargs['grad_scale'] = 10
    else:
        raise ValueError(f"{dataset} dataset is not loaded.")


    lr, arch_lr = (1e-2, 5e-3) if config_kwargs['remain_shape'] else (0.1, 0.05) 

    if arch_default[:3] == 'wrn':
        epochs_default, retrain_epochs = 10, 10
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        sched = [60, 120, 160]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break

            return math.pow(base, optim_factor)

        if dims == 1:
            sched = [30, 60, 90, 120, 160]
        else:
            sched = [60, 120, 160]
        
        def weight_sched_train(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(base, optim_factor)

    return dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs

