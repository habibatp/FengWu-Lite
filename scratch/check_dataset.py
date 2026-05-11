import sys
sys.path.append('.')
import torch
from datasets.era5_npy_f32 import ERA5Dataset
import yaml
import numpy as np

with open('config/fengwu_local_8gb.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset = ERA5Dataset(**config['dataset']['train'])

x, y = dataset[0]
print("x shape:", x.shape, "y shape:", y.shape)
print("x has nan:", torch.isnan(x).any().item())
print("y has nan:", torch.isnan(y).any().item())
print("x min/max:", x.min().item(), x.max().item())
print("y min/max:", y.min().item(), y.max().item())

target = dataset.get_target(0)
print("target shape:", target.shape)
print("target has nan:", np.isnan(target).any().item())
print("target min/max:", np.nanmin(target).item(), np.nanmax(target).item())
