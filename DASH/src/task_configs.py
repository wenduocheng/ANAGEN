import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from itertools import product
from functools import reduce, partial

from networks.deepsea_dash import DeepSEA

from data_loaders import load_deepsea
from task_utils import FocalLoss, LpLoss
from task_utils import auroc


def get_data(dataset, batch_size):
    if dataset == "deepsea": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = load_deepsea(batch_size)
    else:
        raise ValueError(f"{dataset} dataset is not found.")

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    return train_loader, val_loader, test_loader, n_train, n_val, n_test


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
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

        batch_size = 32
        arch_default = 'wrn'  
        config_kwargs['grad_scale'] = 10
    else:
        raise ValueError(f"{dataset} dataset is not loaded.")


    lr, arch_lr = (1e-2, 5e-3) if config_kwargs['remain_shape'] else (0.1, 0.05) 

    if arch_default[:3] == 'wrn':
        epochs_default, retrain_epochs = 100, 200
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

    elif arch_default == 'convnext':
        epochs_default, retrain_epochs, retrain_freq = 100, 300, 100
        opt, arch_opt = torch.optim.AdamW, torch.optim.AdamW
        lr, arch_lr = 4e-3, 1e-2
        weight_decay = 0.05
            
        base_value = lr
        final_value = 1e-6
        niter_per_ep = 392 
        warmup_iters = 0
        epochs = retrain_epochs
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]) / base_value

        def weight_sched_search(iter):
            return schedule[iter]
        
        def weight_sched_train(iter):
            return schedule[iter]

    # arch_opt = ExpGrad

    return dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs


def get_metric(dataset):
    if dataset == "deepsea":
        return auroc, np.max


def get_hp_configs(dataset, n_train):
    epochs = 80
    if n_train < 50:
        subsamping_ratio = 0.2
    elif n_train < 100:
        subsamping_ratio = 0.1
    elif n_train < 500:
        subsamping_ratio = 0.05
    else:
        subsamping_ratio = 0.01

    lrs = 0.1 ** np.arange(1, 4)

    dropout_rates = [0, 0.05]
    wd = [5e-4, 5e-6]
    momentum = [0.9, 0.99]
    configs = list(product(lrs, dropout_rates, wd, momentum))

    return configs, epochs, subsamping_ratio


def get_optimizer(type='SGD', momentum=0.9, weight_decay=5e-4):
    
    return partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, nesterov=(momentum!=0))

