"""
Adapted from: https://github.com/themrzmaster/git-re-basin-pytorch/blob/main/models/resnet.py
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as dist


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('GroupNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.GroupNorm(1, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, planes)

            )


    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_train: int,
        depth: int,
        widen_factor: int,
        dropout_rate: float,
        num_classes: int,
        pool_kernel_size: int = 8,
        prior_scale: float = 1.
    ):
        super(ResNet, self).__init__()
        self.num_train = num_train
        self.prior_scale = prior_scale
        self.in_planes = 16
        self.pool_kernel_size = pool_kernel_size

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(num_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.GroupNorm(1, nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        self.prior_dist = dist.Normal(loc=0., scale=self.prior_scale)


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.pool_kernel_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


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


if __name__ == '__main__':
    num_channels=3
    num_train = 100
    depth = 22
    width = 32
    dropout_rate = 0.
    num_classes = 10
    pool_kernel_size = 8
    prior_scale = 1.
    model = ResNet(num_channels, num_train, depth, width, dropout_rate, num_classes, pool_kernel_size, prior_scale)
    
    from src.data import CIFAR10
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import ModelSummary

    from src.models import MLP
    model = MLP(num_train, input_dim=32*32*3)

    dm = CIFAR10(num_workers=4)
    dm.setup()
    loader = dm.train_dataloader()
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        break
    
    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = model
            self.example_input_array = x
            
        def forward(self, x):
            return self.net(x)

    lit_model = LitModel()

    summary = ModelSummary(lit_model, max_depth=1)
    print(summary)