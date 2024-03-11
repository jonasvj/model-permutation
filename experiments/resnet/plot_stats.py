import os
import hydra
import pandas as pd
import matplotlib.pyplot as plt

"""
ax_to_metric = {
    (0, 0): 'lpd',
    (0, 1): 'acc',
    (1, 0): 'ce',
    (1, 1): 'brier',
}
"""
ax_to_metric = {
    (0): 'lpd',
    (1): 'acc',
    (2): 'ce',
}
metric_to_label ={
    'lpd': 'ELPD',
    'acc': 'Accuracy',
    'ce': 'Calibration error',
    'brier': 'Brier score',
}


@hydra.main(
    config_path='conf/',
    config_name='plot_stats.yaml',
    version_base=None,
)
def main(cfg):
    # Stats
    df_ensemble = pd.read_csv(cfg.stats_ensemble)
    df_dirichlet = pd.read_csv(cfg.stats_dirichlet)

    num_models = len(df_ensemble) - 1
    conc_min = df_dirichlet['concentration'].min()
    conc_max = df_dirichlet['concentration'].max()
    
    # Create plot
    #fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all')
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex='all', figsize=(6.4*3,4.8))

    # Set labels
    for ax_idx, metric in ax_to_metric.items():
        axs[ax_idx].set_ylabel(metric_to_label[metric])
        axs[ax_idx].set_xlabel('Dirichlet concentration')
        axs[ax_idx].set_xscale('log')

    # Plot ensemble and individuals models
    num_models = len(df_ensemble) - 1
    for i in range(num_models + 1):
        ls = '-' if i == num_models else '--'

        # Label
        if i == 0:
            label = 'Models'
        elif i == num_models:
            label = 'Ensemble'
        else:
            label = None

        for ax_idx, metric in ax_to_metric.items():
            axs[ax_idx].hlines(
                y=df_ensemble.iloc[i][metric],
                xmin=conc_min,
                xmax=conc_max,
                color='k',
                linestyle=ls,
                label=label
            )
    
    # Plot Dirichlet ensemble as a function of the concentration
    mean = df_dirichlet.groupby('concentration').mean()
    sem = df_dirichlet.groupby('concentration').sem()
    label = 'Dirichlet ensemble'
    for ax_idx, metric in ax_to_metric.items():
        axs[ax_idx].errorbar(x=mean[metric].index, y=mean[metric], yerr=sem[metric], label=label, fmt='-o')

    first_ax = list(ax_to_metric.keys())[0]
    axs[first_ax].legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.plot_destination, cfg.plot_name))


if __name__ == '__main__':
    main()