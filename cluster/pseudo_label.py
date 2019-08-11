"""
Given folder containing all images, cluster images from pretrained model
and create symbolic links for each cluster

python pseudo_label.py --data tmp --model ../exp32/checkpoint.pth.tar --verbose --exp ./ds32
--cluster-assign ../exp32/clusters
"""
import argparse
import os
import pickle
import time
import sys

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.nn import functional as F


sys.path.append('..')
from util import load_model
import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
from data.downsampled_imagenet import ImageNetDS
from data.constants import DATASET_ROOT
from data.cub200 import CUB200

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--data', type=str, help='path to all dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--cluster-assign', type=str, help='path to final cluster assignment of model')

parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')

parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')

parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')

parser.add_argument('--pct', type=float, default=0.2, help='percentage to sample')


# parser.add_argument('--lr', default=0.05, type=float,
#                     help='learning rate (default: 0.05)')
# parser.add_argument('--wd', default=-5, type=float,
#                     help='weight decay pow (default: -5)')
# parser.add_argument('--reassign', type=float, default=1.,
#                     help="""how many epochs of training between two consecutive
#                     reassignments of clusters (default: 1)""")
# parser.add_argument('--epochs', type=int, default=200,
#                     help='number of total epochs to run (default: 200)')
# parser.add_argument('--start_epoch', default=0, type=int,
#                     help='manual epoch number (useful on restarts) (default: 0)')
# parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to checkpoint (default: None)')
# parser.add_argument('--checkpoints', type=int, default=25000,
#                     help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()
    print(args)

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

    # creating cluster exp
    if not os.path.isdir(args.exp):
        os.makedirs(args.exp)

    print(model)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(32),
           transforms.CenterCrop(32),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    # dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    # dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=True, transform=transforms.Compose(tra))

    # using the target dataset
    dataset = CUB200(root=DATASET_ROOT, train=True, download=True, transform=transforms.Compose(tra))

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # feed the target dataset directly into the model
    model.eval()

    p_cluster = []  # distribution of the cluster from pseudo labels

    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = input_tensor.cuda()
        output = model(input_var)
        p_cl_batch = F.softmax(output * 1000.0, dim=-1)
        p_cluster.extend(p_cl_batch.cpu().detach().numpy())

    p_cluster = np.array(p_cluster)
    print("p_cluster", p_cluster.shape)
    p_c_distn = np.sum(p_cluster, axis=0) / len(dataloader.dataset)

    print("distribution over clusters", p_c_distn)

    # sample from the imagenet dataset with cluster label according to importance weights
    source_dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=True,
                                transform=transforms.ToTensor())

    example_probs = np.zeros(len(source_dataset.train_labels))

    cluster_assignments = pickle.load(open(args.cluster_assign, 'rb'))[0]   # cluster assignments from last epoch

    for c, p_t in enumerate(p_c_distn):
        c_indices = cluster_assignments[c]
        print("sample cluster {} ({} images) with prob {}".format(c, len(c_indices), p_t))
        example_probs[c_indices] = p_c_distn[c] / float(len(c_indices))

    # check the first

    print("Checking the sum should equal to one", np.sum(example_probs))
    example_probs /= example_probs.sum()
    print("After renormalize checking the sume is equal to one", np.sum(example_probs))

    # sample the example indices with/without replacement according to sample size
    sample_size = int(args.pct * len(source_dataset.train_labels))
    sampled_indices = np.random.choice(len(example_probs), sample_size, replace=False, p=example_probs)
    print("sampled {} examples ({})".format(len(sampled_indices), args.pct))

    # print the statistics of sampled
    stats = dict()
    for c in range(1000):
        num_sampled_c = len(np.where(np.array(source_dataset.train_labels)[sampled_indices] == c)[0])
        num_total_c = len(np.where(np.array(source_dataset.train_labels) == c)[0])
        stats[c] = (num_sampled_c, num_total_c)
        print("Sampled {}/{} examples from class {}-{}".format(num_sampled_c, num_total_c, c, source_dataset.label2class[c]))

    save = {'sampled_indices': sampled_indices, 'meta': 'peudo label from deepcluster unsupervised cluster k=10000'}
    save_file = 'deepcluster-sample.pickle'

    with open(save_file, 'wb') as f:
        pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved sampled dataset! {}".format(save_file))

    # clustering algorithm to use
    # deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
    #
    # if os.path.exists(os.path.join(args.exp, 'clusters')):
    #     print("=> loading cluster assignments")
    #     cluster_assignments = pickle.load(open(os.path.join(args.exp, 'clusters'), 'rb'))[0]
    # else:
    #     # cluster the features by computing the pseudo-labels
    #     # 1) remove head
    #     model.top_layer = None
    #     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    #
    #     # 2) get the features for the whole dataset
    #     features = compute_features(dataloader, model, len(dataset))
    #
    #     # 3) cluster the features
    #     print("clustering the features...")
    #     clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
    #
    #     # 4) assign pseudo-labels
    #     cluster_log.log(deepcluster.images_lists)
    #
    #     cluster_assignments = deepcluster.images_lists

    # view_dataset = datasets.ImageFolder(args.data, transform=transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    # ]))

    # view_dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=True,
    #                           transform=torchvision.transforms.ToTensor())
    #
    # # cluster_assignments is a list of len(k), each element corresponds to indices for one cluster
    # for c in range(args.nmb_cluster):
    #     cluster_indices = cluster_assignments[c]
    #     print("cluster {} have {} images".format(c, len(cluster_indices)))
    #     c_dataloader = torch.utils.data.DataLoader(view_dataset, batch_size=64,
    #                                                sampler=SubsetRandomSampler(cluster_indices))
    #
    #     for (images, targets) in c_dataloader:
    #         print("saving cluster {}".format(c), images.shape)
    #         torchvision.utils.save_image(images, os.path.join(args.exp, 'visualize-c{}.png'.format(c)))
    #         break

        # here we want to create a subdir for each cluster inside args.exp
        # create symbolic link from original all dataset to within the subdir
        # for idx in cluster_indices:
        #     print(idx, view_dataset.samples[idx])
        #     break


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            input_var = input_tensor.cuda()
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                print("initializing features ({}, {})".format(N, aux.shape[1]))
                features = np.zeros((N, aux.shape[1])).astype('float32')
                print("initialized features {}".format(features.shape))

            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux.astype('float32')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 10) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    main()
