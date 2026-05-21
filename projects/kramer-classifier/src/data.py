"""Data loading and augmentation for the Kramer classifier."""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool, lighting_robust: bool = False) -> transforms.Compose:
    if train:
        aug = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if lighting_robust:
            # Covers underexposure, overexposure, color temperature shift, and blur/haze.
            # brightness=0.5 → model sees 50–150% of original exposure during training.
            aug += [
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ]
        else:
            aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(aug)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(data_dir: str, batch_size: int = 32, lighting_robust: bool = False):
    """
    Expects data_dir to contain train/ val/ test/ subfolders,
    each with one subfolder per class (ImageFolder format).
    lighting_robust=True applies aggressive lighting augmentation to the training split.
    """
    data_dir = Path(data_dir)
    splits = {}
    for split in ('train', 'val', 'test'):
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Expected {split_path} — run data download first")
        is_train = split == 'train'
        dataset = datasets.ImageFolder(
            split_path,
            transform=get_transforms(is_train, lighting_robust=lighting_robust and is_train),
        )
        splits[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=0,
        )
    return splits['train'], splits['val'], splits['test']
