#!/bin/sh
#BSUB -q gpuv100
#BSUB -J plot_landscape
#BSUB -n 14
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]" 
#BSUB -R "span[hosts=1]"
#BSUB -W 16:00 
#BSUB -o plot_landscape.out 
#BSUB -e plot_landscape.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="resnet/01_mnist"

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_mnist.pdf

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_mnist_permuted.pdf

experiment="resnet/02_fashion_mnist"

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_fashion_mnist.pdf

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_fashion_mnist_permuted.pdf

experiment="resnet/03_cifar10"

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_cifar10.pdf

python3 experiments/resnet/plot_landscape.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet_permuted \
    plot_destination=experiments/${experiment} \
    plot_name=landscape_resnet_cifar10_permuted.pdf