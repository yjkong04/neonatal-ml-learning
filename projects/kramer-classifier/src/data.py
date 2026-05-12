"""Data loading and augmentation for the Kramer classifier."""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(data_dir: str, batch_size: int = 32):
    """
    Expects data_dir to contain train/ val/ test/ subfolders,
    each with one subfolder per class (ImageFolder format).
    """
    data_dir = Path(data_dir)
    splits = {}
    for split in ('train', 'val', 'test'):
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Expected {split_path} — run data download first")
        dataset = datasets.ImageFolder(split_path, transform=get_transforms(split == 'train'))
        splits[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2,
        )
    return splits['train'], splits['val'], splits['test']
