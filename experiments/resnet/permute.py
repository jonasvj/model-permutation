import os
import copy
import torch
import hydra
from src.models import *
from src.utils import model_diff
from src.permutation import weight_match_many
from src.permutation import resnet20_permutation_spec

@hydra.main(
    config_path='conf/',
    config_name='permute.yaml',
    version_base=None,
)
def main(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list()
    model_dicts = list()

    for file in sorted(os.listdir(cfg.model_dir)):
        # Load model
        filepath = os.path.join(cfg.model_dir, file)
        model_dict = torch.load(filepath)

        # Model
        model = eval(model_dict['model_class'])(**model_dict['model_hparams'])
        model.load_state_dict(model_dict['state_dict'])
        
        models.append(model)
        model_dicts.append(model_dict)

    # Copy of models
    new_models = [copy.deepcopy(model) for model in models]
    for i in range(len(new_models)):
        new_models[i] = new_models[i].to(device)

    # Permute models
    permutation_spec = resnet20_permutation_spec()
    new_state_dicts = weight_match_many(
        permutation_spec,
        [model.state_dict() for model in new_models],
        max_iter=cfg.max_iter,
        seed=cfg.seed,
        verbose=True
    )

    # Move back to cpu
    for i in range(len(new_state_dicts)):
        for key in new_state_dicts[i]:
            new_state_dicts[i][key] = new_state_dicts[i][key].to('cpu')
 
    print('Diff. between original model and permuted model.')
    diffs = [
        model_diff(models[i].state_dict(), new_state_dicts[i]) for i in range(len(models))
    ]
    print(diffs)

    save_dir = cfg.model_dir + '_permuted'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i, file in enumerate(sorted(os.listdir(cfg.model_dir))):
        model_dict = model_dicts[i]
        model_dict['state_dict'] = new_state_dicts[i]
        torch.save(model_dict, os.path.join(save_dir, file))


if __name__ == '__main__':
    main()