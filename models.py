# =============================================================================
# models.py - Model Tanimlari
# MobileNetV3-Large, ResNet18, EfficientNet-B0
# =============================================================================

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def get_model(model_name, freeze_backbone=False):
    """
    Secilen modeli yukler ve son katmanini NUM_CLASSES'a gore degistirir.

    Args:
        model_name: "MobileNetV3", "ResNet18", "EfficientNetB0"
        freeze_backbone: True ise backbone katmanlari dondurulur (sadece classifier egitilir)

    Returns:
        model: PyTorch modeli
    """
    if model_name == "MobileNetV3":
        model = models.mobilenet_v3_large(weights="DEFAULT")

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        # Son siniflandirici katmanini degistir
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    elif model_name == "ResNet18":
        model = models.resnet18(weights="DEFAULT")

        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        # Son FC katmanini degistir
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights="DEFAULT")

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        # Son siniflandirici katmanini degistir
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}. "
                         f"Secenekler: MobileNetV3, ResNet18, EfficientNetB0")

    # Egitilecek parametre sayisini hesapla
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model: {model_name}")
    print(f"  Toplam parametre:     {total_params:,}")
    print(f"  Egitilecek parametre: {trainable_params:,}")
    print(f"  Backbone donuk:       {'Evet' if freeze_backbone else 'Hayir'}\n")

    return model


def unfreeze_backbone(model, model_name):
    """Backbone katmanlarini acar (fine-tuning icin)."""

    if model_name == "MobileNetV3":
        for param in model.features.parameters():
            param.requires_grad = True

    elif model_name == "ResNet18":
        for param in model.parameters():
            param.requires_grad = True

    elif model_name == "EfficientNetB0":
        for param in model.features.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{model_name}] Backbone acildi. Egitilecek parametre: {trainable:,}\n")

    return model
