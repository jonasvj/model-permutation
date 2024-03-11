import torch
from typing import Tuple
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.functional import one_hot
from torchmetrics import CalibrationError


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    verbose: bool = True
) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=-1)

            loss += F.nll_loss(log_probs, y, reduction='sum').item()
            correct += log_probs.argmax(dim=-1).eq(y).sum().item()

    loss /= len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)

    if verbose:
        print(f'Loss: {loss:.4f}, Acc.: {acc:.1f}')

    return loss, acc


def evaluate_ensemble(logits, targets):
    # logits:   # S x N x C
    # targets:  #     N
    S, N, C = logits.shape
    eval_stats = dict()

    probs = torch.softmax(logits, dim=-1)                                       # S x N x C
    avg_probs = torch.sum(probs, dim=0) / S                                     #     N x C

    # Predicted class
    class_preds = torch.argmax(avg_probs, dim=-1)                               #     N

    # LPD
    log_densities = dist.Categorical(logits=logits).log_prob(targets)           # S x N   

    eval_stats['lpd'] = (
        -N*torch.log(torch.tensor(S))
        + torch.logsumexp(log_densities, dim=0).sum() 
    ).item() / N

    # Accuracy
    eval_stats['acc'] = torch.sum(
        class_preds == targets
    ).item() / N

    # Calibration error
    ce = CalibrationError(task='multiclass', num_classes=C)
    ce.update(avg_probs, targets)
    eval_stats['ce'] = ce.compute().detach().item()

    # Brier score
    eval_stats['brier'] = ((avg_probs - one_hot(targets))**2).sum().item() / N

    return eval_stats