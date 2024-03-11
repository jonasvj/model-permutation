#!/bin/sh
#BSUB -q hpc
#BSUB -J plot_landscape
#BSUB -n 16 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 24:00 
#BSUB -o plot_landscape.out 
#BSUB -e plot_landscape.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="mlp/01_mnist"
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_mnist.pdf
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_mnist_permuted.pdf

experiment="mlp/02_fashion_mnist"
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_fashion_mnist.pdf
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_fashion_mnist_permuted.pdf

experiment="mlp/03_cifar10"
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_cifar10.pdf
python3 experiments/mlp/plot_landscape.py \
    model_dir=experiments/${experiment}/models_mlp_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_mlp_cifar10_permuted.pdf