import os
import torch
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data import *
from src.models import Ensemble
from src.utils import evaluate_ensemble
from pytorch_lightning import seed_everything


@hydra.main(
    config_path='conf/',
    config_name='compute_stats.yaml',
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dirichlet concentrations
    concentrations = np.geomspace(cfg.conc_min, cfg.conc_max, cfg.num_conc)

    # Ensemble model
    ensemble = Ensemble(model_dir=cfg.model_dir, device=device)

    # Data
    model_dict = torch.load(
        os.path.join(cfg.model_dir, sorted(os.listdir(cfg.model_dir))[0])
    )
    model_cfg = model_dict['cfg']
    dm = eval(model_cfg.data['class'])(**model_cfg.data.hparams)
    dm.setup()
    test_loader = dm.test_dataloader()

    # Targets
    targets = list()
    for _, y in test_loader:
        targets.append(y)
    targets = torch.cat(targets, dim=0)

    # Compute stats ensemble and individaul models
    rows = list()
    num_models = ensemble.num_models
    logits = ensemble.predict_w_ensemble(test_loader)
    for i in range(num_models + 1):
        tmp_logits = logits if i == num_models else logits[i:i+1]
        eval_stats = evaluate_ensemble(tmp_logits, targets)
        rows.append({
            'model': i,
            'lpd': eval_stats['lpd'],
            'acc': eval_stats['acc'],
            'ce': eval_stats['ce'],
            'brier': eval_stats['brier']
        })
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(cfg.stats_destination, cfg.stats_name_ensemble),
        index=False
    )

    # Compute stats for dirichlet ensemble 
    rows = list()
    pbar = tqdm(total=cfg.num_repetitions*len(concentrations)*cfg.num_samples)
    for rep in range(cfg.num_repetitions):
        for conc in concentrations:
            
            logits = list()
            for s in range(cfg.num_samples):
                logits.append(
                    ensemble.predict_w_sample(test_loader, concentration=conc)
                )
                pbar.update(1)
            logits = torch.stack(logits)
            
            eval_stats = evaluate_ensemble(logits, targets)
            rows.append({
                'repetition': rep,
                'concentration': conc,
                'lpd': eval_stats['lpd'],
                'acc': eval_stats['acc'],
                'ce': eval_stats['ce'],
                'brier': eval_stats['brier']
            })
    
    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(cfg.stats_destination, cfg.stats_name_dirichlet),
        index=False
    )


if __name__ == '__main__':
    main()