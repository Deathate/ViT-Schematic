import platform
import random

import torch
import torch.nn.functional as F

import wandb
machine_name = platform.node()
wandb.require("core")
wandb.login(key="dc1e94a79b4faf6ca55ddce9640f3568ef5081a5")
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    name = f"{machine_name}",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        "test":1
    },
)
# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
