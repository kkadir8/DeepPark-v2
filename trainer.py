# =============================================================================
# trainer.py - Egitim Dongusu
# LR Scheduler, Early Stopping, Epoch Metrikleri
# =============================================================================

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import (
    DEVICE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    MODEL_CONFIGS, RESULTS_DIR
)
from models import get_model, unfreeze_backbone


class EarlyStopping:
    """Validation loss iyilesmezse egitimi erken durdurur."""

    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  Early Stopping: {self.patience} epoch boyunca iyilesme yok. Durduruluyor.\n")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Tek bir epoch egitim yapar."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validation/Test seti uzerinde degerlendirme yapar."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model_name, train_loader, val_loader):
    """
    Bir modeli bastan sona egitir.

    Asamalar:
        1. Frozen backbone ile classifier egitimi (freeze_epochs kadar)
        2. Tum katmanlar acik fine-tuning (kalan epoch'lar)
        3. Early stopping ile gereksiz egitimi onleme

    Returns:
        model, history (dict)
    """
    config = MODEL_CONFIGS[model_name]
    freeze_epochs = config["freeze_epochs"]

    print("=" * 60)
    print(f"  MODEL EGITIMI BASLIYOR: {model_name}")
    print("=" * 60)

    # --- Asama 1: Frozen Backbone ---
    print(f"\n--- Asama 1: Backbone Dondurulmus ({freeze_epochs} epoch) ---\n")
    model = get_model(model_name, freeze_backbone=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"]
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    start_time = time.time()

    # Frozen egitim
    for epoch in range(freeze_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- Asama 2: Fine-tuning (Tum katmanlar acik) ---
    remaining_epochs = NUM_EPOCHS - freeze_epochs
    print(f"\n--- Asama 2: Fine-Tuning ({remaining_epochs} epoch) ---\n")

    model = unfreeze_backbone(model, model_name)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"] * 0.1)  # Daha dusuk LR
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler_step"],
        gamma=config["scheduler_gamma"]
    )
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_acc = 0.0

    for epoch in range(remaining_epochs):
        global_epoch = freeze_epochs + epoch

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # En iyi modeli kaydet
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f"{RESULTS_DIR}/{model_name}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            marker = " (*best)"

        print(f"  Epoch [{global_epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")

        # Early stopping kontrolu
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

    duration = time.time() - start_time
    total_epochs = len(history["train_loss"])

    print(f"\n  Egitim Tamamlandi!")
    print(f"  Sure: {duration:.1f} saniye ({duration/60:.1f} dakika)")
    print(f"  Toplam Epoch: {total_epochs}")
    print(f"  En Iyi Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model Kaydedildi: {best_model_path}\n")

    # En iyi modeli yukle
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

    history["duration"] = duration
    history["total_epochs"] = total_epochs
    history["best_val_acc"] = best_val_acc

    return model, history
