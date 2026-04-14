# =============================================================================
# DeepPark v2 - Akilli Otopark Algilama Sistemi
# FET306 Uygulamali Yapay Sinir Aglari - Arasinav Projesi
# Ogrenci: Abdulkadir Gedik
# =============================================================================
# config.py - Tum proje ayarlari tek bir dosyada
# =============================================================================

import os
import torch

# --- Proje Yollari ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"/Users/pc003/Kadir/derin_ogrenme_proje/2012-09-12"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Cihaz Secimi (GPU / MPS / CPU) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Veri Seti Ayarlari ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # macOS uyumlulugu icin
RANDOM_SEED = 42

# Veri bolme oranlari: %70 Train, %15 Validation, %15 Test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet normalizasyon degerleri
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Egitim Ayarlari ---
NUM_EPOCHS = 10
NUM_CLASSES = 2
CLASS_NAMES = ["Empty", "Occupied"]

# Early Stopping
EARLY_STOPPING_PATIENCE = 3

# --- Model Bazli Hiperparametreler ---
MODEL_CONFIGS = {
    "MobileNetV3": {
        "lr": 1e-3,
        "scheduler_step": 3,
        "scheduler_gamma": 0.1,
        "freeze_epochs": 2,    # Ilk 2 epoch backbone dondurulur
    },
    "ResNet18": {
        "lr": 1e-4,
        "scheduler_step": 3,
        "scheduler_gamma": 0.1,
        "freeze_epochs": 2,
    },
    "EfficientNetB0": {
        "lr": 5e-4,
        "scheduler_step": 3,
        "scheduler_gamma": 0.1,
        "freeze_epochs": 2,
    },
}
