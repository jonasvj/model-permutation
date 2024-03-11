import copy
import torch
from typing import List
from src.models import *


def average_models(model_list: List[dict]):
    n_models = len(model_list)
    avg_model = copy.deepcopy(model_list[0])

    params = model_list[0].keys()
    for p in params:
        avg_model[p] = sum(
            [model_list[idx][p] for idx in range(n_models)]
        ) / n_models
    
    return avg_model


def model_diff(model_a_dict: dict, model_b_dict: dict):
    l2_norm = 0.

    params = model_a_dict.keys()
    for p in params:
        l2_norm += torch.sum((model_a_dict[p] - model_b_dict[p])**2)    
    l2_norm = torch.sqrt(l2_norm)

    return l2_norm


def lerp(lam, t1, t2):
    """
    Source:
    https://github.com/themrzmaster/git-re-basin-pytorch/blob/main/utils/utils.py
    """
    t3 = copy.deepcopy(t2)
    for p in t1: 
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def load_model(path, strict=True):
    model_dict = torch.load(path)
    model = eval(model_dict['model_class'])(**model_dict['model_hparams'])
    model.load_state_dict(model_dict['state_dict'], strict=strict)
    
    return model