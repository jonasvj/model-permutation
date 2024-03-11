#!/bin/sh
#BSUB -q hpc
#BSUB -J plot_barriers
#BSUB -n 4 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00 
#BSUB -o plot_barriers.out 
#BSUB -e plot_barriers.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="mlp/01_mnist"
python3 experiments/mlp/plot_barriers.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_mlp_mnist.pdf

experiment="mlp/02_fashion_mnist"
python3 experiments/mlp/plot_barriers.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_mlp_fashion_mnist.pdf

experiment="mlp/03_cifar10"
python3 experiments/mlp/plot_barriers.py \
    model_dir=experiments/${experiment}/models_mlp \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_mlp_cifar10.pdf