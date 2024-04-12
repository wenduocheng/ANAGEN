import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train, tune
import tempfile
from ray.train import Checkpoint

import os
from model import get_model_result
from loader import load_deepsea1
from model_train import train_model
from config import get_config
from model_validate import validate_model
import time


config_set = get_config()
epoch = config_set['epoch']

import psutil
import gc

def free_memory(threshold=20):
    # RAM 
    ram = psutil.virtual_memory()
    ram_percentage = (ram.available / ram.total) * 100
    if ram_percentage < threshold:
        print('Low Ram. Free RAM.')
        gc.collect()
    
    # GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_percentage = (torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / torch.cuda.memory_reserved(0) * 100
        if gpu_percentage < threshold:
            print('Low GPU. Free GPU.')



def train_deepsea(config):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model_result()
    model = model.to(DEVICE)
    # get data
    data = load_deepsea1('path', 32, one_hot = True, valid_split=1,rc_aug=False, shift_aug=False)
    train_loader, valid_loader= data

    optimizer = get_optimizer(model.parameters(), config)
    scheduler = CosineAnnealingLR(optimizer,T_max=20)
    best_loss = float('+inf')

    for i in range(0,epoch):
        start_time = time.time()
        train_loss = train_model(model, optimizer, train_loader, DEVICE)
        valid_loss,auroc,aupr = validate_model(model,valid_loader,DEVICE)
        scheduler.step()
        end_time = time.time()
        run_time = end_time - start_time
        
        free_memory(80)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "best_model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            
            train.report({"train_loss": train_loss,"valid_loss": valid_loss,"auroc":auroc,"aupr":aupr,"epoch_run_time":run_time}, checkpoint=checkpoint)


def get_optimizer(parameters, config):
    if config["optimizer"] == "Adam":
        return optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        return optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
        return optim.SGD(parameters, lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
    elif config["optimizer"] == "NAdam":
        return optim.NAdam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError("Unsupported optimizer")


search_space = {
    "lr": tune.grid_search(config_set['lr']),
    "optimizer": tune.grid_search(config_set['optimizer']),
    "dropout": tune.grid_search(config_set['dropout']),
    "weight_decay": tune.grid_search(config_set['weight_decay'])
}

# Uncomment this to enable distributed execution
# init(address="auto")
asha_scheduler = ASHAScheduler(
    metric="valid_loss",
    mode="min"
)

resources_per_trial = {"cpu": 8, "gpu": 1}

print(f'Start analysis.')
total_time_start = time.time()

runtime_env = {
    'env_vars': {
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_memory_usage_threshold": "0.99"
     }
}

ray.init(runtime_env=runtime_env)
trainable_with_resources = tune.with_resources(train_deepsea, resources_per_trial)
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(scheduler=asha_scheduler),
    param_space=search_space
)
results = tuner.fit()


total_time_end = time.time()
time_in_total = total_time_end-total_time_start

print(f"Analysis done. Total time: {time_in_total}.")


result_file_path = '/home/ec2-user/result.txt'

with open(result_file_path, 'w') as result_file:
    
    best_result = results.get_best_result(metric="valid_loss", mode="min") 
    best_config = best_result.config 
    best_logdir = best_result.path 
    best_checkpoint = best_result.checkpoint  
    best_metrics = best_result.metrics   
    
    result_file.write("Total time of running: {}\n".format(time_in_total))
    result_file.write("Best trial config: {}\n".format(best_config))
    result_file.write("Best trial apth: {}\n".format(best_logdir))
    result_file.write("Best trial metric: {}\n".format(best_metrics))
    

    for path, df in {result.path: result.metrics_dataframe for result in results}.items():
        result_file.write("\nMetrics for trial {}:\n".format(path))
        df_str = df.to_string(header=True, index=True)
        result_file.write(df_str)
        result_file.write("\n") 

    print("Results saved to {}".format(result_file_path))

