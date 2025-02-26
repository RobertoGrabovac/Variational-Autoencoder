from typing import List, Optional, Sequence, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA # our dataset is already supported in torch


class CustomCelebA(CelebA):
    """
    A workaround for the default CelebA dataset class in PyTorch. Sometimes, 
    the default implementation checks dataset integrity using _check_integrity, 
    which might fail due to dataset structure or downloading issues. This 
    class overrides _check_integrity to always return True, but assumes that
    dataset is already downloaded.
    """
    def _check_integrity(self) -> bool:
        return True


class Dataset(LightningDataModule):
    """
    LightningDataModule is a class in PyTorch Lightning that provides a standardized interface
    for handling data in machine learning workflows. It helps manage data loading, preprocessing,
    and splitting in a clean and organized manner. It also supports the CelebA dataset.

    Args:
        data_dir (str): Root directory of the dataset.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        test_batch_size (int): Batch size for testing.
        center_crop (int): Size of the center crop.
        patch_size (Union[int, Sequence[int]]): Size of the crop for image resizing.
        pin_memory (bool): Whether to load data into pinned memory for faster transfer to the GPU.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        center_crop: int,
        patch_size: Union[int, Sequence[int]],
        pin_memory: bool = False, # enables fast data transfer to the GPU if True as it is located in RAM
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.center_crop = center_crop
        self.patch_size = patch_size
        self.num_workers = 4
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.center_crop),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])
        
        # CelebA provides a file named list_eval_partition.txt, which specifies
        # which images go into which partition (train, validation, test). For 
        # example:  000001.jpg  0
        #           000002.jpg  1
        #           000003.jpg  2
        #           000004.jpg  0
        #           000005.jpg  1
        # The number corresponds to: 0 for the train split, 1 for the validation
        # split and 2 for the test split.
        self.train_dataset = CustomCelebA(
            self.data_dir,
            split='train',
            transform=data_transforms,
            download=False,
        )
        
        self.val_dataset = CustomCelebA(
            self.data_dir,
            split='valid',
            transform=data_transforms,
            download=False,
        )

        self.test_dataset = CustomCelebA(
            self.data_dir,
            split='test',
            transform=data_transforms,
            download=False,
        )
        
        
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True, # to prevent order biases and improve model generalization
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False, # to ensure consistent and fair evaluation of model performance during validation
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )