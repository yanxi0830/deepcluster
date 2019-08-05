# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python cluster_features.py /scratch/ssd001/datasets/imagenet/val --exp ./out/ --arch alexnet --k
# 10 --sobel --verbose --resume ../deepcluster_models/alexnet/checkpoint.pth.tar
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

sys.path.append('..')
import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
from data.downsampled_imagenet import ImageNetDS
from data.constants import DATASET_ROOT

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16', 'alexnet32'], default='alexnet32',
                    help='CNN architecture (default: alexnet32)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
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

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # model.top_layer = nn.Linear(fd, args.nmb_cluster)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in list(checkpoint['state_dict'].keys()):
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
                    print("deleting {}".format(key))
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
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
    dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=True, transform=transforms.Compose(tra))

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    if os.path.exists(os.path.join(args.exp, 'clusters')):
        print("=> loading cluster assignments")
        cluster_assignments = pickle.load(open(os.path.join(args.exp, 'clusters'), 'rb'))[0]
    else:
        # cluster the features by computing the pseudo-labels
        # 1) remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # 2) get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))

        # 3) cluster the features
        print("clustering the features...")
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # 4) assign pseudo-labels
        cluster_log.log(deepcluster.images_lists)

        cluster_assignments = deepcluster.images_lists

    # view_dataset = datasets.ImageFolder(args.data, transform=transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    # ]))

    view_dataset = ImageNetDS(DATASET_ROOT + 'downsampled-imagenet-32/', 32, train=True,
                              transform=torchvision.transforms.ToTensor())

    for c in range(args.nmb_cluster):
        cluster_indices = cluster_assignments[c]
        print("cluster {} have {} images".format(c, len(cluster_indices)))
        c_dataloader = torch.utils.data.DataLoader(view_dataset, batch_size=64,
                                                   sampler=SubsetRandomSampler(cluster_indices))

        for (images, targets) in c_dataloader:
            print("saving cluster {}".format(c), images.shape)
            torchvision.utils.save_image(images, os.path.join(args.exp, 'hahah{}.png'.format(c)))
            break

    # # training convnet with DeepCluster
    # for epoch in range(args.start_epoch, args.epochs):
    #     end = time.time()
    #
    #     # remove head
    #     model.top_layer = None
    #     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    #
    #     # get the features for the whole dataset
    #     features = compute_features(dataloader, model, len(dataset))
    #
    #     # cluster the features
    #     clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
    #
    #     # assign pseudo-labels
    #     train_dataset = clustering.cluster_assign(deepcluster.images_lists,
    #                                               dataset.imgs)
    #
    #     # uniformely sample per target
    #     sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
    #                                deepcluster.images_lists)
    #
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch,
    #         num_workers=args.workers,
    #         sampler=sampler,
    #         pin_memory=True,
    #     )
    #
    #     # set last fully connected layer
    #     mlp = list(model.classifier.children())
    #     mlp.append(nn.ReLU(inplace=True).cuda())
    #     model.classifier = nn.Sequential(*mlp)
    #     model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
    #     model.top_layer.weight.data.normal_(0, 0.01)
    #     model.top_layer.bias.data.zero_()
    #     model.top_layer.cuda()
    #
    #     # train network with clusters as pseudo-labels
    #     end = time.time()
    #     loss = train(train_dataloader, model, criterion, optimizer, epoch)
    #
    #     # print log
    #     if args.verbose:
    #         print('###### Epoch [{0}] ###### \n'
    #               'Time: {1:.3f} s\n'
    #               'Clustering loss: {2:.3f} \n'
    #               'ConvNet loss: {3:.3f}'
    #               .format(epoch, time.time() - end, clustering_loss, loss))
    #         try:
    #             nmi = normalized_mutual_info_score(
    #                 clustering.arrange_clustering(deepcluster.images_lists),
    #                 clustering.arrange_clustering(cluster_log.data[-1])
    #             )
    #             print('NMI against previous assignment: {0:.3f}'.format(nmi))
    #         except IndexError:
    #             pass
    #         print('####################### \n')
    #     # save running checkpoint
    #     torch.save({'epoch': epoch + 1,
    #                 'arch': args.arch,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer': optimizer.state_dict()},
    #                os.path.join(args.exp, 'checkpoint.pth.tar'))
    #
    #     # save cluster assignments
    #     cluster_log.log(deepcluster.images_lists)


# def train(loader, model, crit, opt, epoch):
#     """Training of the CNN.
#         Args:
#             loader (torch.utils.data.DataLoader): Data loader
#             model (nn.Module): CNN
#             crit (torch.nn): loss
#             opt (torch.optim.SGD): optimizer for every parameters with True
#                                    requires_grad in model except top layer
#             epoch (int)
#     """
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     data_time = AverageMeter()
#     forward_time = AverageMeter()
#     backward_time = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     # create an optimizer for the last fc layer
#     optimizer_tl = torch.optim.SGD(
#         model.top_layer.parameters(),
#         lr=args.lr,
#         weight_decay=10 ** args.wd,
#     )
#
#     end = time.time()
#     for i, (input_tensor, target) in enumerate(loader):
#         data_time.update(time.time() - end)
#
#         # save checkpoint
#         n = len(loader) * epoch + i
#         if n % args.checkpoints == 0:
#             path = os.path.join(
#                 args.exp,
#                 'checkpoints',
#                 'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
#             )
#             if args.verbose:
#                 print('Save checkpoint at: {0}'.format(path))
#             torch.save({
#                 'epoch': epoch + 1,
#                 'arch': args.arch,
#                 'state_dict': model.state_dict(),
#                 'optimizer': opt.state_dict()
#             }, path)
#
#         target = target.cuda()
#         input_var = torch.autograd.Variable(input_tensor.cuda())
#         target_var = torch.autograd.Variable(target)
#
#         output = model(input_var)
#         loss = crit(output, target_var)
#
#         # record loss
#         losses.update(loss.item(), input_tensor.size(0))
#
#         # compute gradient and do SGD step
#         opt.zero_grad()
#         optimizer_tl.zero_grad()
#         loss.backward()
#         opt.step()
#         optimizer_tl.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if args.verbose and (i % 200) == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss: {loss.val:.4f} ({loss.avg:.4f})'
#                   .format(epoch, i, len(loader), batch_time=batch_time,
#                           data_time=data_time, loss=losses))
#
#     return losses.avg


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            # input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
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
