# =============================================================================
# dataset.py - Veri Yukleme ve On Isleme
# Train / Validation / Test bolmesi (Stratified)
# =============================================================================

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import (
    DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO,
    IMAGENET_MEAN, IMAGENET_STD
)


def get_transforms():
    """Egitim ve test icin veri donusumlerini dondurur."""

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_test_transform


def load_datasets():
    """
    PKLot veri setini yukler ve stratified olarak 3'e boler.
    Returns: train_loader, val_loader, test_loader, class_names
    """
    train_transform, val_test_transform = get_transforms()

    # Ayni veri setini farkli transform'larla yukle
    full_data_train = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    full_data_eval = datasets.ImageFolder(root=DATASET_PATH, transform=val_test_transform)

    targets = full_data_train.targets
    class_names = full_data_train.classes
    indices = np.arange(len(targets))

    # 1. Adim: Train (%70) ve Gecici (%30) olarak bol
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_RATIO + 0.15),
        shuffle=True, stratify=targets, random_state=RANDOM_SEED
    )

    # 2. Adim: Gecici seti Validation (%15) ve Test (%15) olarak bol
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        shuffle=True, stratify=temp_targets, random_state=RANDOM_SEED
    )

    # Subset olustur
    train_dataset = Subset(full_data_train, train_idx)
    val_dataset = Subset(full_data_eval, val_idx)
    test_dataset = Subset(full_data_eval, test_idx)

    # DataLoader olustur
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    print(f"Veri Seti Dagilimi:")
    print(f"  Train:      {len(train_dataset):,} goruntu")
    print(f"  Validation: {len(val_dataset):,} goruntu")
    print(f"  Test:       {len(test_dataset):,} goruntu")
    print(f"  Toplam:     {len(train_dataset) + len(val_dataset) + len(test_dataset):,} goruntu")
    print(f"  Siniflar:   {class_names}")
    print(f"  Cihaz:      Yuklenmeye hazir\n")

    return train_loader, val_loader, test_loader, class_names
