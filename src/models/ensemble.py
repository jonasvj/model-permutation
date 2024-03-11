import os
import copy
import torch
import torch.nn as nn
from typing import Optional
from src.models import MLP, ResNet
import torch.distributions as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Ensemble(nn.Module):
    def __init__(self, model_dir: str, device: str = 'cpu'):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        
        self.locs = None
        self.model = None
        self.num_models = None
        self.load_models()


    def load_models(self):
        self.locs = list()
        for file in sorted(os.listdir(self.model_dir)):
            model_dict = torch.load(os.path.join(self.model_dir, file))

            # Model
            model = eval(model_dict['model_class'])(**model_dict['model_hparams'])
            model.load_state_dict(model_dict['state_dict'])
            model = model.to('cpu')
            self.locs.append(parameters_to_vector(model.parameters()))

        self.num_models = len(self.locs)
        self.locs = torch.stack(self.locs)

        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)


    def sample_convex_comb(self, concentrations: torch.Tensor):
        concentrations = torch.clamp(concentrations, min=2e-3)
        weights = dist.Dirichlet(concentration=concentrations).sample()
        mean = self.locs.T @ weights
        return mean


    def predict_w_sample(
        self,
        dataloader: torch.utils.data.DataLoader,
        sample: Optional[torch.Tensor] = None,
        concentration: float = 1.
    ):
        self.model.eval()

        with torch.no_grad():
            if sample is None:
                conc = concentration*torch.ones(self.num_models)
                sample = self.sample_convex_comb(conc)

            sample = sample.to(self.device)
            vector_to_parameters(sample, self.model.parameters())

            logits = list()
            for x, _ in dataloader:
                x = x.to(self.device)
                logits.append(self.model(x).detach().to('cpu'))
            logits = torch.cat(logits, dim=0)

            return logits


    def predict_w_ensemble(self, dataloader):
        logits = list()
        for param_vector in self.locs:
            logits_i = self.predict_w_sample(dataloader, sample=param_vector)
            logits.append(logits_i)

        return torch.stack(logits)