import torch
import hydra
from src.data import *
from tqdm import trange
from src.models import *
from pytorch_lightning import seed_everything


@hydra.main(
    config_path='conf/',
    config_name='config.yaml',
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize data
    dm = eval(cfg.data['class'])(**cfg.data.hparams)
    dm.setup()
    num_train = len(dm.data_train)
    num_batches = len(dm.train_dataloader())

    # Initialize model
    model = eval(cfg.model['class'])(num_train=num_train, **cfg.model.hparams)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Data loader
    data_loader = dm.train_dataloader()
    
    pbar = trange(cfg.num_epochs)
    for epoch in pbar:
        avg_loss = 0.
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss = model.loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_step = loss.detach().item()
            avg_loss += loss_step

        avg_loss /= num_batches
        pbar.set_postfix_str(f'loss_epoch={avg_loss:.4e}')

    # Save model
    model.to('cpu')
    torch.save(
        {
            'model_class': cfg.model['class'],
            'model_hparams': {'num_train': num_train, **cfg.model.hparams},
            'state_dict': model.state_dict(),
            'cfg': cfg
        },
        f"{cfg.model_dir}/{cfg.model_name}"
    )

if __name__ == '__main__':
    main()
