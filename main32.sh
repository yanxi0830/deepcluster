# Train Unsupervised Experts
# bash main32.sh PICKLE_FILE OUTPUT_DIR
#!/bin/bash

set -x

for (( e=0; e<10; e++ ))
do
    EXP=$2/c$e;
    mkdir -p ${EXP};
    python main32.py --exp ${EXP} --arch alexnet32 --sobel --verbose --workers 32 --cluster-file $1 --cluster-idx $e --k 3000;
done
