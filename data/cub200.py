import os
import pandas as pd
import numpy as np
import sys

import torch
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision


class CUB200(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # save 32x32 image to file to save time
        # downsample_trans = transforms.Resize(32)
        #
        # for idx, filepath in enumerate(self.filepaths):
        #     orig_path = os.path.join(self.root, self.base_folder, filepath)
        #     new_path = os.path.join(self.root, 'CUB_200_2011/images32', filepath)
        #     orig_img = self.loader(orig_path)
        #     ds_img = downsample_trans(orig_img)
        #
        #     try:
        #         img = Image.open(orig_path)
        #     except Exception:
        #         print("saving", ds_img, new_path)
        #         if not os.path.exists(os.path.dirname(new_path)):
        #             os.makedirs(os.path.dirname(new_path))
        #         ds_img.save(new_path)
        # exit(1)

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.targets = np.array([self.data.iloc[idx].target - 1 for idx in range(len(self.data))])
        self.filepaths = np.array([self.data.iloc[idx].filepath for idx in range(len(self.data))])

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        # for index, row in self.data.iterrows():
        #     filepath = os.path.join(self.root, self.base_folder, row.filepath)
        #     if not os.path.isfile(filepath):
        #         print(filepath)
        #         return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # sample = self.data.iloc[idx]
        # path = os.path.join(self.root, self.base_folder, sample.filepath)
        # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        # img = self.loader(path)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # return img, target
        path = os.path.join(self.root, self.base_folder, self.filepaths[idx])
        target = self.targets[idx]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CUB200Subset(CUB200):
    """
    CUB200 with train/val/test split
    """

    def __init__(self, root, train_val_test, transform=None, loader=default_loader, download=True):
        if train_val_test == 'train' or train_val_test == 'val':
            train = True
        else:
            train = False

        super(CUB200Subset, self).__init__(root=root, train=train, transform=transform,
                                           loader=default_loader, download=download)

        # randomly split training and validation into 0.9 train / 0.1 val
        indices = list(range(len(self.data)))
        split = int(0.1 * len(self.data))
        np.random.seed(1)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        if train_val_test == 'train':
            print("0.9/0.1 train split")
            self.data = self.data.iloc[train_indices]
            self.targets = self.targets[train_indices]
            self.filepaths = self.filepaths[train_indices]
        elif train_val_test == 'val':
            print("0.9/0.1 val split")
            self.data = self.data.iloc[val_indices]
            self.targets = self.targets[val_indices]
            self.filepaths = self.filepaths[val_indices]


class CUB200SubsetLimited(CUB200):
    """
    Using only limited target images to select the source sample, this gives the limited
    # wait...can we just use the index?
    """

    def __init__(self, root, train=True, transform=None, download=True, num_images_per_class=0):
        super(CUB200SubsetLimited, self).__init__(root=root, train=train, transform=transform,
                                                  loader=default_loader, download=download)

        self.num_images_per_class = num_images_per_class
        if num_images_per_class > 0:
            print("Selecting {} images per class".format(num_images_per_class))
            limited_indices = self.get_indices_for_num_images_per_class(num_images_per_class)
            print(limited_indices)
            self.targets = np.array(self.targets)[limited_indices]
            self.filepaths = self.filepaths[limited_indices]

    def get_indices_for_num_images_per_class(self, num_images_per_class):
        """
        1) Iterate through the targets, add indices for each class as long as not saturated
        Return indices of selected limited subset
        """
        class2indices = {}
        res = []
        for idx in range(len(self.targets)):
            cls = self.targets[idx]
            if cls not in class2indices:
                class2indices[cls] = []
            if len(class2indices[cls]) < num_images_per_class:
                res.append(idx)
            class2indices[cls].append(idx)

        return res
