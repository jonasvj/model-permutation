import torch
import pytorch_lightning as pl
from torchvision import datasets
from typing import Optional, Tuple, Sequence
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Subset


class FashionMNIST(pl.LightningDataModule):
    """"
    The Fashion-MNIST dataset wrapped in a PyTorch Lightning data module.
    """
    def __init__(
        self,
        data_dir: str = 'data/',
        batch_size_train: int = 128,
        batch_size_inference: int = 128,
        data_augmentation: bool = False,
        num_workers: int = 0,
        num_val: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.data_augmentation = data_augmentation
        self.num_workers = num_workers
        self.num_val = num_val
        self.seed = seed

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self) -> None:
        """
        Downloads the data.
        """
        datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)
    

    def split_data(
        self,
        num_train: int = 60000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits training indices into train and validation indices.
        """
        indices = torch.randperm(
            num_train,
            generator=torch.Generator().manual_seed(self.seed)
        )
        val_indices = indices[:self.num_val]
        train_indices = indices[self.num_val:]

        return train_indices, val_indices


    def compute_channel_stats(
        self,
        train_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mean and standard deviation of each channel in the 
        training images.
        """
        data_train = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ]),
        )
        data_train = Subset(data_train, indices=train_indices)
        
        X_train = torch.stack([img[0] for img in data_train])
        mean = X_train.mean(dim=(0,2,3))
        std = X_train.std(dim=(0,2,3))

        return mean, std


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Does all the necessary data preprocessing.
        """
        # Split training data into training and validation data
        train_indices, val_indices = self.split_data(num_train=60000)
        # Compute channel means and standard deviations for training data
        mean, std = self.compute_channel_stats(train_indices=train_indices)

        # Transformation for training and testing
        test_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
        if self.data_augmentation:
            train_transforms = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean, std),
            ]
        else:
            train_transforms = test_transforms[:]

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        self.data_train = Subset(
            datasets.FashionMNIST(
                root=self.data_dir, train=True, transform=train_transforms
            ),
            indices=train_indices,
        )
        self.data_val = Subset(
            datasets.FashionMNIST(
                root=self.data_dir, train=True, transform=test_transforms
            ),
            indices=val_indices,
        )
        self.data_test = datasets.FashionMNIST(
            root=self.data_dir, train=False, transform=test_transforms
        )


    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

if __name__ == '__main__':
    dm = FashionMNIST(data_augmentation=True)
    dm.prepare_data()
    dm.setup()