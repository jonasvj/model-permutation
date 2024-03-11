import numpy as np
from typing import List
from src.utils import average_models, model_diff
from src.permutation.weight_matching import ( weight_matching, PermutationSpec,
    apply_permutation )


def weight_match_many(
    perm_spec: PermutationSpec,
    model_list: List[dict],
    max_iter: int = 100,
    seed: int = 0,
    verbose=True
):
    """
    Implementation of the MergeMany algorithm from "Git Re-Basin: Merging
    Models modulo Permutation Symmetries" (https://arxiv.org/abs/2209.04836).
    """
    n_models = len(model_list)
    rng = np.random.default_rng(seed)
    
    for iter in range(max_iter):
        progress = False

        random_model_order = rng.permutation(n_models)
        for model_idx in random_model_order:
            # Select model and average the others
            selected_model = model_list[model_idx]
            other_models = [
                model_list[idx] for idx in range(n_models) if idx != model_idx
            ]
            avg_model = average_models(other_models)

            # For checking convergence
            l2_before = model_diff(avg_model, selected_model)

            # Permute model 
            permutation = weight_matching(
                perm_spec,
                avg_model,
                selected_model,
                max_iter=100,
                verbose=True
            )
            new_model = apply_permutation(perm_spec, permutation, selected_model)
            model_list[model_idx] = new_model

            # Check convergence 
            l2_after = model_diff(avg_model, new_model)
            progress = progress or l2_after < l2_before - 1e-12
            
            if verbose:
                print(
                    f'Merge Many iteration: {iter:02}, '
                    f'Progress: {l2_before - l2_after:.4f}'
                )

        if not progress:
            break

    return model_list