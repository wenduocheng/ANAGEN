import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os

class ConvNet(nn.Module):
    def __init__(self, dropout_rate):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
def train_mnist(config):
    # Data Setup
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
        batch_size=64, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet(dropout_rate=config["dropout"]).to(device)

    optimizer = get_optimizer(model.parameters(), config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)

    for epoch in range(config["epochs"]):
        train_loss = train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)
        scheduler.step()

        # Report metrics for Ray Tune
        tune.report(loss=train_loss, accuracy=acc)

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
    "lr": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
    "optimizer": tune.grid_search(["Adam", "AdamW", "SGD", "NAdam"]),
    "dropout": tune.grid_search([0, 0.05, 0.1, 0.2]),
    "epochs": tune.grid_search([20, 40, 60, 80]),
    "weight_decay": tune.grid_search([1e-4, 1e-5])
}

scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=max(search_space["epochs"]),
    reduction_factor=2,
    grace_period=min(search_space["epochs"])
)

reporter = CLIReporter(
    metric_columns=["loss", "accuracy", "training_iteration"]
)

# Uncomment this to enable distributed execution
ray.init(address="auto")

results = tune.run(
    train_mnist,
    resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
    config=search_space,
    num_samples=1,
    scheduler=scheduler,
    progress_reporter=reporter
)
