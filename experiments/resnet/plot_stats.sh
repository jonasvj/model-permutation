#!/bin/sh
#BSUB -q hpc
#BSUB -J plot_stats
#BSUB -n 1 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00 
#BSUB -o plot_stats.out 
#BSUB -e plot_stats.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="resnet/01_mnist"
python3 experiments/resnet/plot_stats.py \
    stats_ensemble=experiments/${experiment}/stats_resnet_mnist_permuted_ensemble.csv \
    stats_dirichlet=experiments/${experiment}/stats_resnet_mnist_permuted_dirichlet.csv \
    plot_destination=experiments/${experiment} \
    plot_name=stats_resnet_mnist_permuted.pdf

experiment="resnet/02_fashion_mnist"
python3 experiments/resnet/plot_stats.py \
    stats_ensemble=experiments/${experiment}/stats_resnet_fashion_mnist_permuted_ensemble.csv \
    stats_dirichlet=experiments/${experiment}/stats_resnet_fashion_mnist_permuted_dirichlet.csv \
    plot_destination=experiments/${experiment} \
    plot_name=stats_resnet_fashion_mnist_permuted.pdf


experiment="resnet/03_cifar10"
python3 experiments/resnet/plot_stats.py \
    stats_ensemble=experiments/${experiment}/stats_resnet_cifar10_permuted_ensemble.csv \
    stats_dirichlet=experiments/${experiment}/stats_resnet_cifar10_permuted_dirichlet.csv \
    plot_destination=experiments/${experiment} \
    plot_name=stats_resnet_cifar10_permuted.pdf