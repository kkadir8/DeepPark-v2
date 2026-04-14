# =============================================================================
# demo.py - Sunum Demosu
# 3 Modelin tahminlerini yan yana gosterir
# =============================================================================

import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, RESULTS_DIR, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from dataset import load_datasets
from models import get_model


def denormalize(tensor):
    """ImageNet normalizasyonunu geri alir."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def load_trained_models():
    """Egitilmis 3 modeli yukler."""
    model_names = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
    loaded = {}

    for name in model_names:
        model = get_model(name, freeze_backbone=False)
        path = f"{RESULTS_DIR}/{name}_best.pth"
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        model.eval()
        loaded[name] = model
        print(f"  {name} yuklendi.")

    return loaded


def run_demo(num_samples=4):
    """
    Rastgele test goruntuleri secip 3 modelin tahminlerini
    yan yana gosterir. Sunum icin ideal.
    """
    print("\n" + "=" * 50)
    print("  DEEPPARK v2 - CANLI DEMO")
    print("=" * 50 + "\n")

    # Veri ve modelleri yukle
    _, _, test_loader, class_names = load_datasets()
    models_dict = load_trained_models()

    # Test setinden rastgele ornek al
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    indices = random.sample(range(len(all_images)), num_samples)

    # Gorsellesstirme
    model_names = list(models_dict.keys())
    fig, axes = plt.subplots(num_samples, len(model_names) + 1, figsize=(16, 4 * num_samples))

    for row, idx in enumerate(indices):
        image = all_images[idx]
        true_label = all_labels[idx].item()

        # Orijinal goruntu
        ax = axes[row, 0]
        img_display = denormalize(image).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        ax.set_title(f"Gercek: {class_names[true_label]}",
                     fontsize=11, fontweight="bold", color="black")
        ax.axis("off")
        if row == 0:
            ax.set_title(f"ORIJINAL\nGercek: {class_names[true_label]}",
                         fontsize=11, fontweight="bold")

        # Her model icin tahmin
        for col, (model_name, model) in enumerate(models_dict.items(), 1):
            ax = axes[row, col]
            ax.imshow(img_display)

            with torch.no_grad():
                inp = image.unsqueeze(0).to(DEVICE)
                output = model(inp)
                prob = F.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

            predicted_label = class_names[pred.item()]
            confidence = conf.item() * 100
            is_correct = pred.item() == true_label

            color = "green" if is_correct else "red"
            status = "Dogru" if is_correct else "YANLIS!"
            title = f"{predicted_label} (%{confidence:.1f})\n{status}"

            if row == 0:
                title = f"{model_name}\n{title}"

            ax.set_title(title, fontsize=10, fontweight="bold", color=color)
            ax.axis("off")

    plt.suptitle("DeepPark v2 - Model Tahmin Karsilastirmasi",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/demo_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n  Demo kaydedildi: {RESULTS_DIR}/demo_predictions.png")


if __name__ == "__main__":
    run_demo(num_samples=4)
