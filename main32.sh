# Train Unsupervised Experts
# bash main32.sh PICKLE_FILE
#!/bin/bash

set -x

for (( e=0; e<10; e++ ))
do
    EXP=./iic-head0-model5000-e60/c$e;
    mkdir -p ${EXP};
    python main32.py --exp ${EXP} --arch alexnet32 --verbose --workers 32 --cluster-file $1 --cluster-idx $e --k 3000;
done
