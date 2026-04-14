# =============================================================================
# DeepPark v2 - Akilli Otopark Algilama Sistemi
# FET306 Uygulamali Yapay Sinir Aglari - Arasinav Projesi
# Ogrenci: Abdulkadir Gedik
# =============================================================================
# main.py - Ana Giris Noktasi
#
# Kullanim:
#   python main.py              -> Tum modelleri egitir ve degerlendirir
#   python main.py --evaluate   -> Sadece kayitli modelleri degerlendirir
#   python main.py --demo       -> Sunum demosu calistirir
# =============================================================================

import sys
import torch
import numpy as np
import random

from config import DEVICE, RESULTS_DIR, RANDOM_SEED
from dataset import load_datasets
from models import get_model
from trainer import train_model
from evaluate import evaluate_model, generate_all_plots


def set_seed(seed):
    """Tekrarlanabilirlik icin tum rastgelelik kaynaklarini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header():
    """Proje basligi."""
    print("\n" + "=" * 60)
    print("   DeepPark v2 - Akilli Otopark Algilama Sistemi")
    print("   FET306 Uygulamali Yapay Sinir Aglari")
    print("   Abdulkadir Gedik")
    print(f"   Cihaz: {DEVICE}")
    print("=" * 60 + "\n")


def full_pipeline():
    """Tam egitim ve degerlendirme pipeline'i."""
    print_header()
    set_seed(RANDOM_SEED)

    # 1. Veri setini yukle
    print("[1/4] Veri seti yukleniyor...\n")
    train_loader, val_loader, test_loader, class_names = load_datasets()

    # 2. Modelleri egit
    print("[2/4] Modeller egitiliyor...\n")
    model_names = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
    trained_models = {}
    all_histories = {}

    for name in model_names:
        model, history = train_model(name, train_loader, val_loader)
        trained_models[name] = model
        all_histories[name] = history

    # 3. Test seti uzerinde degerlendir
    print("[3/4] Modeller degerlendiriliyor...\n")
    all_results = {}

    for name, model in trained_models.items():
        labels, preds, probs, metrics = evaluate_model(model, name, test_loader)
        all_results[name] = {
            "labels": labels,
            "preds": preds,
            "probs": probs,
            "metrics": metrics,
        }

    # 4. Grafikleri olustur
    print("[4/4] Grafikler olusturuluyor...\n")
    generate_all_plots(all_results, all_histories)

    # Ozet
    print("\n" + "=" * 60)
    print("  SONUC OZETI")
    print("=" * 60)
    for name in model_names:
        m = all_results[name]["metrics"]
        h = all_histories[name]
        print(f"  {name:20s} | Acc: {m['accuracy']*100:.2f}% | "
              f"F1: {m['f1_weighted']:.4f} | AUC: {m['roc_auc']:.4f} | "
              f"Sure: {h['duration']:.0f}s")
    print("=" * 60)
    print(f"  Tum sonuclar '{RESULTS_DIR}/' klasorunde.\n")


def evaluate_only():
    """Sadece kayitli modelleri yukleyip degerlendirir."""
    print_header()
    set_seed(RANDOM_SEED)

    _, _, test_loader, class_names = load_datasets()
    model_names = ["MobileNetV3", "ResNet18", "EfficientNetB0"]

    all_results = {}
    all_histories = {}

    for name in model_names:
        model = get_model(name, freeze_backbone=False)
        path = f"{RESULTS_DIR}/{name}_best.pth"
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        labels, preds, probs, metrics = evaluate_model(model, name, test_loader)
        all_results[name] = {
            "labels": labels, "preds": preds,
            "probs": probs, "metrics": metrics
        }
        # Bos history (grafikler icin)
        all_histories[name] = {"duration": 0}

    # Confusion matrix + ROC curve
    from evaluate import plot_confusion_matrix, plot_roc_curves
    for name, data in all_results.items():
        plot_confusion_matrix(data["labels"], data["preds"], name)
    plot_roc_curves(all_results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--evaluate":
            evaluate_only()
        elif sys.argv[1] == "--demo":
            from demo import run_demo
            run_demo()
        else:
            print("Kullanim: python main.py [--evaluate | --demo]")
    else:
        full_pipeline()
