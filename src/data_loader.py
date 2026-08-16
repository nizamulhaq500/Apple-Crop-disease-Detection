from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "split"


def get_dataloaders(
    split_dir_path=DEFAULT_SPLIT,
    image_size=128,
    batch_size=32,
    num_workers=0
):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test splits.

    Returns:
        train_loader, val_loader, test_loader, classes (list of class names)
    """
    split_dir = Path(split_dir_path)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            "Did you run data_split.py?"
        )

    # 1. Define Transforms
    # We augment the training data to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # We DO NOT augment validation and test data
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # 2. Create Dataset objects
    # ImageFolder expects subdirectories to be class names
    train_dataset = datasets.ImageFolder(
        split_dir / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        split_dir / "val", transform=evaluation_transform
    )
    test_dataset = datasets.ImageFolder(
        split_dir / "test", transform=evaluation_transform
    )

    # 3. Create DataLoaders
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    print("Testing data loaders...")
    train_dl, val_dl, test_dl, classes = get_dataloaders()

    print(f"Classes: {classes}")
    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print(f"Test batches: {len(test_dl)}")

    # Let's inspect the first batch to verify tensor shapes
    images, labels = next(iter(train_dl))
    print(
        f"Batch images shape: {images.shape} "
        "(batch_size, channels, height, width)"
    )
    print(f"Batch labels shape: {labels.shape}")
