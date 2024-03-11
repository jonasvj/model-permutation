#!/bin/sh
#BSUB -q hpc
#BSUB -J compute_stats
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00 
#BSUB -o compute_stats.out 
#BSUB -e compute_stats.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="mlp/01_mnist"
python3 experiments/mlp/compute_stats.py \
    model_dir=experiments/${experiment}/models_mlp \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_mlp_mnist_ensemble.csv \
    stats_name_dirichlet=stats_mlp_mnist_dirichlet.csv


python3 experiments/mlp/compute_stats.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_mlp_mnist_permuted_ensemble.csv \
    stats_name_dirichlet=stats_mlp_mnist_permuted_dirichlet.csv

experiment="mlp/02_fashion_mnist"
python3 experiments/mlp/compute_stats.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_mlp_fashion_mnist_permuted_ensemble.csv \
    stats_name_dirichlet=stats_mlp_fashion_mnist_permuted_dirichlet.csv

experiment="mlp/03_cifar10"
python3 experiments/mlp/compute_stats.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    stats_destination=experiments/${experiment} \
    stats_name_ensemble=stats_mlp_cifar10_permuted_ensemble.csv \
    stats_name_dirichlet=stats_mlp_cifar10_permuted_dirichlet.csv