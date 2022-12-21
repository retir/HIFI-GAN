import torch
import numpy as np

from torch.utils.data import DataLoader
from datasets.collate_fn import collate_func
from datasets.datasets import BaseDataset


def get_dataloaders(config):
    train_dataset = BaseDataset(config['data_path'], config['train_split'])
    val_dataset = BaseDataset(config['data_path'], config['val_split'])

    training_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_func,
        drop_last=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_func,
        num_workers=8
    )
    return training_loader, val_loader