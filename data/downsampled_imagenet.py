import os
import numpy as np
import pickle

import torch
import torch.utils.data as data
import torchvision.transforms
from PIL import Image
from torchvision.datasets.utils import check_integrity


class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = np.array(self.train_labels)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = np.array(self.test_labels)

        # load map_clsloc.txt mapping label index to class names
        self.label2class = dict()
        self.class2label = dict()
        with open(os.path.join(self.root, 'map_clsloc.txt'), 'rb') as map_f:
            for line in map_f.readlines():
                _, idx, name = line.split()
                self.label2class[int(idx) - 1] = str(name, 'utf-8')
                self.class2label[str(name, 'utf-8')] = int(idx) - 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def get_indices_for_class(self, class_label, num_indices):
        """
        Return num_indices of random indices of images with class_label
        """
        all_indices = np.where(np.array(self.train_labels) == class_label)[0]
        np.random.seed(1)
        res = np.random.choice(all_indices, num_indices, replace=False)
        return res

    def get_subset_indices(self):
        subset_path = os.path.join(self.root, 'moe_fixed_subset{}.npy'.format(50))
        if os.path.exists(subset_path):
            subset_indices = np.load(subset_path)
            return subset_indices
        else:
            return []


class ImageNetDSPseudoLabel(ImageNetDS):
    def __init__(self, root, img_size, train=True, transform=None, target_transform=None, images_lists=None):
        super(ImageNetDSPseudoLabel, self).__init__(root, img_size, train=train, transform=transform,
                                                    target_transform=target_transform)
        # image_list is a list with k list that contains indices of targets
        for cluster, images in enumerate(images_lists):
            self.train_labels[images] = cluster


class ImageNetDSKMeans(ImageNetDS):
    """
    Given cluster file containing cluster labels for each example, and cluster idx
    Construct the dataset of the specified cluster
    """

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None,
                 cluster_file='', cluster_idx=0, stats=False):
        super(ImageNetDSKMeans, self).__init__(root, img_size, train=train, transform=transform,
                                               target_transform=target_transform)
        if not train:
            return

        cluster_indices = []
        with open(cluster_file, 'rb') as f:
            d = pickle.load(f)
            labels = d['label']
            if isinstance(cluster_idx, list):
                for c in cluster_idx:
                    cluster_indices.extend(np.where(labels == c)[0])
            else:
                cluster_indices.extend(np.where(labels == cluster_idx)[0])

            self.train_data = self.train_data[cluster_indices]
            print("Loaded from {}, using cluster {}, {}".format(cluster_file, cluster_idx, self.train_data.shape))
            self.train_labels = np.array(self.train_labels)[cluster_indices]

        if stats:
            # print statistics of the cluster (how many images in each class?)
            total = len(self.train_labels)
            class2count = []
            for idx in self.label2class:
                count = len(np.where(self.train_labels == idx)[0])
                pct = float(count) / float(total)
                print("{} {} - cluster have {}/{} images ({})".format(idx, self.label2class[idx], count, total, pct))
                class2count.append(count)
            # print top 5 categories
            top5ind = np.argpartition(np.array(class2count), -5)[-5:]
            print([(self.label2class[i], class2count[i]) for i in top5ind])


class ImageNetDSPseudoLabelSubset(ImageNetDSKMeans):
    def __init__(self, root, img_size, train=True, transform=None, target_transform=None, images_lists=None,
                 cluster_file='', cluster_idx=0):
        super(ImageNetDSPseudoLabelSubset, self).__init__(root, img_size, train=train, transform=transform,
                                                          target_transform=target_transform,
                                                          cluster_file=cluster_file, cluster_idx=cluster_idx)

        # image_list is a list with k list that contains indices of targets
        for cluster, images in enumerate(images_lists):
            self.train_labels[images] = cluster

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)