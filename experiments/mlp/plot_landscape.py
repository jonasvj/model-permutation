import os
import torch
import hydra
from src.data import *
from copy import deepcopy
import matplotlib.pyplot as plt
from src.utils import load_model
from pytorch_lightning import seed_everything
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class NegLogJoint:
    def __init__(self, num_points=30, model_dir='models/mnist_mlp', device='cuda'):
        self.num_points = num_points
        self.model_dir = model_dir
        self.device = device

        # Model paths
        model_paths = [
            os.path.join(self.model_dir, model)
            for model in sorted(os.listdir(self.model_dir))
        ]
        model_paths = model_paths[:3]

        # Models
        models = list()
        for path in model_paths:
            models.append(load_model(path))

        self.model = deepcopy(models[0])
        self.model_params = torch.stack(
            [parameters_to_vector(m.parameters()) for m in models]
        ).T.detach()

        # Orthonormal basis for the 2D-plane containing w1, w2, w3
        # w_1 is the origin and u, v are the basis vectors
        self.w_1 = self.model_params[:,0]
        self.u = self.model_params[:,1] - self.w_1
        self.v = self.model_params[:,2] - self.w_1
        
        # Orthogonalize u and v
        self.c = torch.dot(self.v, self.u) / torch.dot(self.u, self.u)
        self.v = self.v - self.c*self.u
        
        # Normalize u and v
        self.unorm = torch.linalg.vector_norm(self.u)
        self.vnorm = torch.linalg.vector_norm(self.v)
        self.uhat = self.u / self.unorm
        self.vhat = self.v / self.vnorm
        
        # Define grid to evaluate density on
        self.x = torch.linspace(-0.2*self.unorm.item(), 1.2*self.unorm.item(), self.num_points)
        self.y = torch.linspace(-0.2*self.vnorm.item(), 1.2*self.vnorm.item(), self.num_points)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='xy')
    
        model_dict = torch.load(model_paths[0])
        model_cfg = model_dict['cfg']
        dm = eval(model_cfg.data['class'])(**model_cfg.data.hparams)
        dm.setup()
        self.test_loader = dm.test_dataloader()


    def neg_log_joint_grid(self, llh=False):
        grid_points = torch.concat(
            [self.X.reshape(-1, 1), self.Y.reshape(-1, 1)],
            dim=1
        )
        neg_log_joint = torch.zeros(len(grid_points))

        for i, point in enumerate(grid_points):
            print(f'Evaluating point {i+1:03}/{len(grid_points)}')
            
            vec = self.point_to_vec(point)
            neg_log_joint[i] = self.neg_log_joint(vec, return_llh=llh)

        return neg_log_joint.reshape(self.num_points, self.num_points)

    
    def point_to_vec(self, point):
        x, y = point
        vec = self.w_1 + x*self.uhat + y*self.vhat

        return vec


    def neg_log_joint(self, vec, return_llh=False):
        vec = vec.to(self.device)
        vector_to_parameters(vec, self.model.parameters())

        with torch.no_grad():
            llh = 0.
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                llh += self.model.log_likelihood(x, y)
            
            if return_llh:
                log_joint = llh
            else:
                log_joint = llh + self.model.log_prior()

            return -(log_joint / len(self.test_loader.dataset)).detach().item()


    def plot_neg_log_joint_grid(self, figure_path, llh=False, num_levels=30):
        fig, ax = plt.subplots()

        neg_log_joint_grid = self.neg_log_joint_grid(llh=llh)

        cnt = ax.contourf(
            self.X,
            self.Y,
            neg_log_joint_grid,
            levels=num_levels,
        )
        
        # Create axis for colorbar
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes('right', size='5%', pad=0.2)
        fig.add_axes(ax_cb)

        # Create colorbar
        plt.colorbar(cnt, cax=ax_cb)

        # Plot points
        ax.scatter(0, 0, c='k')
        ax.scatter(self.unorm, 0, c='k')
        ax.scatter(self.c*self.unorm, self.vnorm, c='k')

        ax.set_aspect('equal') 
        plt.tight_layout()
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close()


@hydra.main(
    config_path='conf/',
    config_name='plot_landscape.yaml',
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    neg_log_joint = NegLogJoint(
        num_points=cfg.num_points,
        model_dir=cfg.model_dir,
        device=device,
    )
    neg_log_joint.plot_neg_log_joint_grid(
        figure_path=os.path.join(cfg.plot_destination, cfg.plot_name),
        llh=cfg.llh,
        num_levels=cfg.num_levels,
    )


if __name__ == '__main__':
    main()