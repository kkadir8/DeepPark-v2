# =============================================================================
# evaluate.py - Kapsamli Model Degerlendirme
# Confusion Matrix, ROC Curve, Classification Report, Model Karsilastirma
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix
)

from config import DEVICE, RESULTS_DIR, CLASS_NAMES


def predict(model, test_loader):
    """Model ile tahmin yapar, etiketleri ve olasiliklari dondurur."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    """Tum performans metriklerini hesaplar."""
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
    except Exception:
        metrics["roc_auc"] = 0.0

    return metrics


def evaluate_model(model, model_name, test_loader):
    """Tek bir modeli degerlendirir ve sonuclari yazdirir."""
    labels, preds, probs = predict(model, test_loader)
    metrics = compute_metrics(labels, preds, probs)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Test Sonuclari")
    print(f"{'='*50}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (W):     {metrics['f1_weighted']:.4f}")
    print(f"  F1 (M):     {metrics['f1_macro']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES)}")

    return labels, preds, probs, metrics


# =========================================================================
# GORSELLLESTIRME FONKSIYONLARI
# =========================================================================

def plot_confusion_matrix(labels, preds, model_name):
    """Confusion matrix heatmap cizer."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 16, "fontweight": "bold"}
    )
    acc = accuracy_score(labels, preds) * 100
    plt.title(f"{model_name} - Confusion Matrix\n(Accuracy: %{acc:.2f})",
              fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Gercek Deger", fontsize=11)
    plt.xlabel("Tahmin Edilen", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"  Kaydedildi: {model_name}_confusion_matrix.png")


def plot_roc_curves(all_results):
    """Tum modellerin ROC egrilerini tek grafikte cizer."""
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, (model_name, data) in enumerate(all_results.items()):
        labels = data["labels"]
        probs = data["probs"][:, 1]
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                 label=f"{model_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Rastgele (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve Karsilastirmasi", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curve_comparison.png", dpi=300)
    plt.close()
    print(f"  Kaydedildi: roc_curve_comparison.png")


def plot_training_history(all_histories):
    """Tum modellerin egitim gecmisini karsilastirir (Loss + Accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"MobileNetV3": "#2196F3", "ResNet18": "#FF5722", "EfficientNetB0": "#4CAF50"}

    for model_name, history in all_histories.items():
        c = colors[model_name]
        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss grafigi
        axes[0].plot(epochs, history["train_loss"], color=c, linestyle="-",
                     linewidth=2, label=f"{model_name} Train")
        axes[0].plot(epochs, history["val_loss"], color=c, linestyle="--",
                     linewidth=2, label=f"{model_name} Val", alpha=0.7)

        # Accuracy grafigi
        axes[1].plot(epochs, history["train_acc"], color=c, linestyle="-",
                     linewidth=2, label=f"{model_name} Train")
        axes[1].plot(epochs, history["val_acc"], color=c, linestyle="--",
                     linewidth=2, label=f"{model_name} Val", alpha=0.7)

    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=300)
    plt.close()
    print(f"  Kaydedildi: training_history.png")


def plot_model_comparison_table(all_results, all_histories):
    """Model karsilastirma tablosunu gorsel olarak olusturur."""
    model_names = list(all_results.keys())

    # Tablo verileri
    rows = []
    for name in model_names:
        m = all_results[name]["metrics"]
        h = all_histories[name]

        # Model boyutu (MB)
        model_path = f"{RESULTS_DIR}/{name}_best.pth"
        size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0

        rows.append([
            name,
            f"{m['accuracy']*100:.2f}%",
            f"{m['f1_weighted']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{h['duration']:.1f}s",
            f"{size_mb:.1f} MB",
        ])

    columns = ["Model", "Accuracy", "F1 (W)", "Precision", "Recall", "ROC AUC", "Sure", "Boyut"]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Baslik satiri renklendirme
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Satir renkleri
    row_colors = ["#E3F2FD", "#FBE9E7", "#E8F5E9"]
    for i in range(len(rows)):
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(row_colors[i])

    plt.title("Model Karsilastirma Tablosu", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: model_comparison_table.png")


def generate_all_plots(all_results, all_histories):
    """Tum grafikleri tek seferde olusturur."""
    print(f"\n{'='*50}")
    print("  GRAFIKLER OLUSTURULUYOR")
    print(f"{'='*50}\n")

    # Her model icin confusion matrix
    for model_name, data in all_results.items():
        plot_confusion_matrix(data["labels"], data["preds"], model_name)

    # ROC curve karsilastirmasi
    plot_roc_curves(all_results)

    # Egitim gecmisi
    plot_training_history(all_histories)

    # Karsilastirma tablosu
    plot_model_comparison_table(all_results, all_histories)

    print(f"\n  Tum grafikler '{RESULTS_DIR}/' klasorune kaydedildi.\n")
