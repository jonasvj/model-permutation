"""
Adapted from: https://github.com/themrzmaster/git-re-basin-pytorch/blob/main/models/mlp.py.
"""
import torch
import torch.nn as nn
import torch.distributions as dist


class MLP(nn.Module):
    def __init__(
        self,
        num_train: int,
        input_dim: int = 28*28,
        prior_scale: float = 1.,
    ):
        super().__init__()
        self.num_train = num_train
        self.input_dim = input_dim
        self.prior_scale = prior_scale

        self.layer0 = nn.Linear(self.input_dim, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 10)

        self.prior_dist = dist.Normal(loc=0., scale=self.prior_scale)


    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_dim)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)

        return x


    def log_prior(self, param_vector=None):
        if param_vector is not None:
            log_prior = self.prior_dist.log_prob(param_vector).sum()
        else:
            log_prior = 0.
            for p in self.parameters():
                log_prior += self.prior_dist.log_prob(p).sum()

        return log_prior


    def log_density(self, x: torch.Tensor, y: torch.Tensor):
        logits = self(x).squeeze()
        y = y.squeeze()
        return dist.Categorical(logits=logits).log_prob(y)


    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor):
       return self.log_density(x, y).sum()
    
    
    def log_joint(self, x: torch.tensor, y: torch.Tensor):
        mb_factor = self.num_train / len(y)
        return mb_factor*self.log_likelihood(x, y) + self.log_prior()
    

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        return -self.log_joint(x, y)
