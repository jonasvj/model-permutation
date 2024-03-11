#!/bin/sh
#BSUB -q hpc
#BSUB -J permute
#BSUB -n 4 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00 
#BSUB -o permute.out 
#BSUB -e permute.err 

source ~/.virtualenvs/perm_gmm/bin/activate
cd ~/model-permutation

experiment="mlp/01_mnist"
python3 experiments/mlp/permute.py \
    model_dir=experiments/${experiment}/models_mlp

experiment="mlp/02_fashion_mnist"
python3 experiments/mlp/permute.py \
    model_dir=experiments/${experiment}/models_mlp

experiment="mlp/03_cifar10"
python3 experiments/mlp/permute.py \
    model_dir=experiments/${experiment}/models_mlp