# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/scratch/ssd001/datasets/imagenet/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
EXP="./exp"

mkdir -p ${EXP}

python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
--lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
