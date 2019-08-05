import torch

DATASET_ROOT = '/h/yanxi/Disk/datasets/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
