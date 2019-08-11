from __future__ import print_function

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from PIL import Image
import scipy
import scipy.io
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

sys.path.append('../')
from torchvision.datasets.folder import default_loader


class OxfordPets(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        self.img_path2class = {}
        self.data = []
        self.targets = []
        anno_path = 'trainval.txt' if self.train else 'test.txt'
        with open(os.path.join(self.root, 'annotations/{}'.format(anno_path))) as f:
            lines = f.readlines()
            for l in lines:
                image_path, class_id, _, _ = l.split()
                image_path = os.path.join(os.path.join(self.root, 'images'), image_path) + '.jpg'
                self.img_path2class[image_path] = int(class_id) - 1
                self.data.append(image_path)
                self.targets.append(int(class_id) - 1)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx], self.targets[idx]

        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class OxfordPetsSplit(OxfordPets):
    """
    StanfordCars with train/val/test split
    """

    def __init__(self, root, train_val_test, transform=None):
        if train_val_test == 'train' or train_val_test == 'val':
            train = True
        else:
            train = False

        super(OxfordPetsSplit, self).__init__(root, train=train, transform=transform)

        # randomly split training and validation into 0.9 train / 0.1 val
        indices = list(range(len(self.targets)))
        split = int(0.2 * len(self.targets))
        np.random.seed(1)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        if train_val_test == 'train':
            print("0.8/0.2 train split")
            self.data = self.data[train_indices]
            self.targets = self.targets[train_indices]
        elif train_val_test == 'val':
            print("0.8/0.2 val split")
            self.data = self.data[val_indices]
            self.targets = self.targets[val_indices]


# if __name__ == "__main__":
#     transform_test = transforms.Compose([
#         transforms.Resize(64),
#         transforms.CenterCrop(64),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     train_dataset = OxfordPetsSplit(DATASET_ROOT + 'OxfordIIITPet', train_val_test='val', transform=transform_test)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
#
#     print("num images: {}".format(len(train_dataset)))
#
#     n_classes = len(np.unique(train_dataset.targets))
#     print("n_classes ", n_classes)
#     for iteration, (images, targets) in enumerate(train_loader):
#         print("Iteration ", iteration)
#         print(targets)
#         torchvision.utils.save_image(images, 'checkpets.png')
#         break
