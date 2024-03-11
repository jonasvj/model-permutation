#!/bin/sh
#BSUB -q gpuv100
#BSUB -J compute_stats
#BSUB -n 14
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]" 
#BSUB -R "span[hosts=1]"
#BSUB -W 14:00 
#BSUB -o compute_stats_resnet.out 
#BSUB -e compute_stats_resnet.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="resnet/01_mnist"

python3 experiments/resnet/compute_stats.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_resnet_mnist_permuted_ensemble.csv \
    stats_name_dirichlet=stats_resnet_mnist_permuted_dirichlet.csv

experiment="resnet/02_fashion_mnist"

python3 experiments/resnet/compute_stats.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_resnet_fashion_mnist_permuted_ensemble.csv \
    stats_name_dirichlet=stats_resnet_fashion_mnist_permuted_dirichlet.csv

experiment="resnet/03_cifar10"

python3 experiments/resnet/compute_stats.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_resnet_cifar10_permuted_ensemble.csv \
    stats_name_dirichlet=stats_resnet_cifar10_permuted_dirichlet.csv