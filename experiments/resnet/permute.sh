#!/bin/sh
#BSUB -q gpuv100
#BSUB -J permute
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00 
#BSUB -o permute.out 
#BSUB -e permute.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="resnet/01_mnist"
python3 experiments/resnet/permute.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet

experiment="resnet/02_fashion_mnist"
python3 experiments/resnet/permute.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet

experiment="resnet/03_cifar10"
python3 experiments/resnet/permute.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet