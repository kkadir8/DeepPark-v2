# DeepPark v2 - Akilli Otopark Algilama Sistemi

**FET306 Uygulamali Yapay Sinir Aglari - Arasinav Projesi**

**Ogrenci:** Abdulkadir Gedik (23040301069)

**Bolum:** Yazilim Muhendisligi - Istanbul Topkapi Universitesi

---

## Proje Hakkinda

Guvenlik kamerasi goruntulerinden otopark alanlarinin bos veya dolu oldugunu tespit eden derin ogrenme tabanli bir siniflandirma sistemidir. PKLot veri seti uzerinde uc farkli transfer ogrenme modeli karsilastirilmistir.

## Modeller ve Sonuclar

| Model | Accuracy | F1 Score | ROC AUC | Sure | Boyut |
|---|---|---|---|---|---|
| MobileNetV3-Large | %99.98 | 0.9998 | 1.0000 | 23.7 dk | 16.2 MB |
| ResNet18 | %100.00 | 1.0000 | 1.0000 | 31.5 dk | 42.7 MB |
| EfficientNet-B0 | %99.96 | 0.9996 | 1.0000 | 40.5 dk | 15.6 MB |

## Veri Seti

- **PKLot Dataset**: 32,327 goruntu (14,032 Empty + 18,295 Occupied)
- **Split**: %70 Train / %15 Validation / %15 Test (Stratified)

## Kullanim

```bash
# Ortam kurulumu
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Tam egitim pipeline
python main.py

# Sadece degerlendirme (onceden egitilmis modellerle)
python main.py --evaluate

# Sunum demosu
python main.py --demo
```

## Proje Yapisi

```
├── config.py          # Merkezi ayarlar
├── dataset.py         # Veri yukleme (Train/Val/Test)
├── models.py          # 3 model tanimi
├── trainer.py         # Egitim (LR Scheduler + Early Stopping)
├── evaluate.py        # Metrikler + Grafikler
├── demo.py            # Canli sunum demosu
├── main.py            # Ana giris noktasi
├── requirements.txt   # Bagimliliklar
└── results/           # Model agirliklari + grafikler
```

## Teknik Detaylar

- **Iki asamali egitim**: Backbone dondurma (2 epoch) + Fine-tuning (8 epoch'a kadar)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **LR Scheduler**: StepLR (step=3, gamma=0.1)
- **Early Stopping**: patience=3
- **Data Augmentation**: Flip, Rotation, ColorJitter, Affine
