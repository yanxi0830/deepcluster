# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
import sys

import pickle
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('..')
from util import AverageMeter, learning_rate_decay, load_model, Logger
from data.downsampled_imagenet import ImageNetDS
from data.constants import DATASET_ROOT
import clustering

parser = argparse.ArgumentParser(description="""Run KMeans clustering on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5], default=5,
                    help='on top of which convolutional layer extract features')
parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--num_cluster', '--k', type=int, default=10,
                    help='number of cluster for k-means (default: 10)')


def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    model = load_model(args.model)
    model.cuda()
    cudnn.benchmark = True

    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=False,
                         transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)
    # freeze batch norm layers
    model.eval()

    # all_features = compute_features(data_loader, model, len(dataset))
    # print("all_features", all_features.shape, all_features)
    #
    # deepcluster = clustering.Kmeans(args.num_cluster)
    # clustering_loss = deepcluster.cluster(all_features, verbose=args.verbose)
    #
    # print(deepcluster.images_lists)

    all_features = []

    for i, (images, target) in enumerate(tqdm(data_loader)):
        input_var = images.cuda()
        features = forward(input_var, model, args.conv).detach().cpu().numpy()
        # print("conv {} features".format(args.conv), features.shape)
        all_features.extend(features)

    all_features = np.array(all_features)
    pca_features = preprocess_features(all_features)
    print("pca features", pca_features.shape)

    # kmeans on features
    # print("computing kmeans cluster...", pca_features.shape)
    # kmeans = KMeans(n_clusters=args.num_cluster, random_state=0, verbose=args.verbose).fit(pca_features)
    # cluster_labels = np.array(kmeans.labels_)
    # print("finished kmeans")

    print("faiss kmeans...", pca_features.shape)
    I, loss = run_kmeans(pca_features, args.num_cluster, verbose=args.verbose)

    print(I)

    filename = 'deepcluster-k{}-conv{}-cluster.pickle'.format(args.num_cluster, args.conv)
    save = {'label': cluster_labels}
    with open(filename, 'wb') as f:
        pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved kmeans cluster to {}".format(save))

    # visualize the clustering
    view_dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=False,
                              transform=torchvision.transforms.ToTensor())

    for c in range(args.num_cluster):
        cluster_indices = np.where(cluster_labels == c)[0]
        print("cluster {} have {} images".format(c, len(cluster_indices)))
        c_dataloader = torch.utils.data.DataLoader(view_dataset, batch_size=64,
                                                   sampler=SubsetRandomSampler(cluster_indices))

        for (images, targets) in c_dataloader:
            print("saving cluster {}".format(c), images.shape)
            torchvision.utils.save_image(images, 'cluster{}-conv{}.png'.format(c, args.conv))
            break


def forward(x, model, conv):
    """
    Given input and pretrained alexnet, forward the features specified by conv layer
    """
    if conv == 1:
        av_pool = nn.AvgPool2d(6, stride=6, padding=3)
        s = 9600
    elif conv == 2:
        av_pool = nn.AvgPool2d(4, stride=4, padding=0)
        s = 9216
    elif conv == 3:
        av_pool = nn.AvgPool2d(3, stride=3, padding=1)
        s = 9600
    elif conv == 4:
        av_pool = nn.AvgPool2d(3, stride=3, padding=1)
        s = 9600
    elif conv == 5:
        av_pool = nn.AvgPool2d(2, stride=2, padding=0)
        s = 9216
    else:
        av_pool = nn.Sequential()

    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                x = av_pool(x)
                x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
                return x
            count = count + 1

    x = av_pool(x)
    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    return x


def compute_features(dataloader, model, N):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(tqdm(dataloader)):
        input_var = input_tensor.cuda()
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            features[i * args.batch_size: (i + 1) * args.batch_size] = aux.astype('float32')
        else:
            # special treatment for final batch
            features[i * args.batch_size:] = aux.astype('float32')

    return features


def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


if __name__ == '__main__':
    main()
