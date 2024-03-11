import os
import copy
import torch
import hydra
import string
import numpy as np
from src.data import *
import matplotlib.pyplot as plt
from itertools import combinations
from src.utils import lerp, evaluate, load_model


def plot_interp_acc(
    ax,
    lambdas,
    train_loss_interp_naive,
    test_loss_interp_naive,
    train_loss_interp_clever,
    test_loss_interp_clever,
    legend=True,
):
    ax.plot(
        lambdas,
        train_loss_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=1,
        label="Train, naïve interp."
    )
    ax.plot(
        lambdas,
        test_loss_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=1,
        label="Test, naïve interp."
    )
    ax.plot(
        lambdas,
        train_loss_interp_clever,
        linestyle="solid",
        color="tab:blue",
        linewidth=1,
        label="Train, permuted interp."
    )
    ax.plot(
        lambdas,
        test_loss_interp_clever,
        linestyle="solid",
        color="tab:orange",
        linewidth=1,
        label="Test, permuted interp."
    )
    if legend:
        ax.legend(loc="lower right", framealpha=0.5)
    legend_handles, legend_lables = ax.get_legend_handles_labels()

    return legend_handles, legend_lables


def plot_main(
    ax,
    train_loader,
    test_loader,
    model_a_path,
    model_b_path,
    model_a_perm_path,
    model_b_perm_path,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambdas = torch.linspace(0, 1, steps=25)

    test_loss_interp_clever = []
    test_loss_interp_naive = []
    train_loss_interp_clever = []
    train_loss_interp_naive = []

    # naive
    model_a = load_model(model_a_path)
    model_b = load_model(model_b_path)

    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())

    for lam in lambdas:
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, _ = evaluate(model_b, test_loader, device, verbose=False)
      test_loss_interp_naive.append(test_loss)
      train_loss, _ = evaluate(model_b, train_loader, device, verbose=False)
      train_loss_interp_naive.append(train_loss)

    # smart
    model_a = load_model(model_a_perm_path)
    model_b = load_model(model_b_perm_path)

    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())

    for lam in lambdas:
      smart_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(smart_p)
      test_loss, _ = evaluate(model_b, test_loader, device, verbose=False)
      test_loss_interp_clever.append(test_loss)
      train_loss, _ = evaluate(model_b, train_loader, device, verbose=False)
      train_loss_interp_clever.append(train_loss)

    legend_handles, legend_lables = plot_interp_acc(
        ax,
        lambdas,
        train_loss_interp_naive,
        test_loss_interp_naive,
        train_loss_interp_clever,
        test_loss_interp_clever,
        legend=False
    )
    return legend_handles, legend_lables


@hydra.main(
    config_path='conf/',
    config_name='plot_barriers.yaml',
    version_base=None,
)
def main(cfg): 
    model_paths_orig = [
       os.path.join(cfg.model_dir, model) 
       for model in sorted(os.listdir(cfg.model_dir))
    ]
    model_paths_perm = [
       os.path.join(cfg.model_dir + '_permuted', model) 
       for model in sorted(os.listdir(cfg.model_dir + '_permuted'))
    ]
    model_names = list(string.ascii_letters[:len(model_paths_orig)])
    
    model_pairs_orig = list(combinations(model_paths_orig, 2))
    model_pairs_perm = list(combinations(model_paths_perm, 2))
    model_pairs_names = list(combinations(model_names, 2))

    # Data
    model_dict = torch.load(model_paths_orig[0])
    model_cfg = model_dict['cfg']
    dm = eval(model_cfg.data['class'])(**model_cfg.data.hparams)
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    # Plot
    num_models = len(model_names)
    fig, ax = plt.subplots(
        nrows=num_models - 1,
        ncols=num_models - 1,
        sharey='all',
    )
    gs = ax[0,0].get_gridspec()

    indices_upper = np.stack(np.triu_indices(num_models, k=1), axis=1)
    for i in range(len(model_pairs_orig)):
        row, col = indices_upper[i]
        col -= 1

        model_a, model_b = model_pairs_orig[i]
        model_a_perm, model_b_perm = model_pairs_perm[i]
        model_a_name, model_b_name = model_pairs_names[i]
        print(f'Plotting model pair: {i+1}/{len(model_pairs_orig)}')
        print(f'Model 1            : {model_a}')
        print(f'Model 2            : {model_b}')
        print(f'Model 1 permuted   : {model_a_perm}')
        print(f'Model 2 permuted   : {model_b_perm}')

        ax[row, col].set_xticks([0, 1])
        ax[row, col].set_xticklabels([model_a_name, model_b_name])
        ax[row,col].yaxis.set_tick_params(which='both', labelbottom=False, bottom=False)
        
        legend_handles, legend_lables = plot_main(
            ax[row, col],
            train_loader,
            test_loader,
            model_a,
            model_b,
            model_a_perm,
            model_b_perm,
        )

    indices_lower = np.stack(np.tril_indices(num_models - 1, k=-1), axis=1)
    for (row, col) in indices_lower:
        ax[row, col].set_axis_off()

    ax[2,0].remove()
    ax[2,1].remove()
    ax[3,0].remove()
    ax[3,1].remove()
    axbig = fig.add_subplot(gs[2:, :2])
    axbig.legend(
        legend_handles,
        legend_lables, 
        loc='lower left',
    )
    axbig.set_axis_off()

    fig.tight_layout()
    fig.savefig(
        os.path.join(cfg.plot_destination, cfg.plot_name),
        bbox_inches='tight'
    )

if __name__ == '__main__':
    main()