#!/bin/sh
#BSUB -q gpuv100
#BSUB -J plot_barriers
#BSUB -n 14
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]" 
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00 
#BSUB -o plot_barriers_resnet.out 
#BSUB -e plot_barriers_resnet.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="resnet/01_mnist"
python3 experiments/resnet/plot_barriers.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_resnet_mnist.pdf

experiment="resnet/02_fashion_mnist"
python3 experiments/resnet/plot_barriers.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_resnet_fashion_mnist.pdf

experiment="resnet/03_cifar10"
python3 experiments/resnet/plot_barriers.py \
    model_dir=/work3/jovje/model-permutation/experiments/${experiment}/models_resnet \
    plot_destination=experiments/${experiment} \
    plot_name=barriers_resnet_cifar10.pdf