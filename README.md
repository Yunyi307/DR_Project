# Diabetic Retinopathy Detection & Grading with Domain Generalisation

MSc project (EEE8097, Newcastle University). Deep-learning system that grades
diabetic retinopathy (DR) severity (0-4) from retinal fundus photographs, compares
CNN vs. Vision-Transformer backbones, and studies **cross-dataset generalisation**.



## Setup

```bash
pip install -r requirements.txt
```

## 1. Get the data (one-time Kaggle auth)

The APTOS 2019 set is a Kaggle competition dataset. The signed URL originally
supplied has expired, so use the Kaggle API:

1. Accept the competition rules (free Kaggle account required):
   https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
2. Kaggle -> Settings -> API -> *Create New Token* -> save `kaggle.json` to
   `C:\Users\<you>\.kaggle\kaggle.json`.
3. Download & unpack:
   ```bash
   python scripts/download_data.py
   ```

## 2. Create the split and preprocessing cache

```bash
python scripts/prepare_splits.py                     # data/splits.csv + class EDA
python scripts/cache_preprocess.py --variant full    # 512px cache (crop+CLAHE+Ben Graham)
python scripts/cache_preprocess.py --variant raw     # crop-only cache (for the E4 ablation)
```

The cache is required before training: on-the-fly Ben Graham on full-res images
starves the GPU, so we render each image once (~1 min for all 3,662).

## 3. Train

```bash
python train.py --config configs/effnet_b4.yaml     # E1 CNN baseline (Focal)
python train.py --config configs/vit_b16.yaml       # E3 Vision Transformer
python train.py --config configs/swin_b.yaml        # E3 Swin Transformer
python train.py --loss ce  --name effnet_b4_ce      # E2 loss ablation
python train.py --loss weighted_ce --name effnet_b4_wce
```

Results (checkpoints, metrics, history) are written to `outputs/<name>/`.

## 4. Evaluate, interpret, generalise

```bash
python scripts/make_figures.py effnet_b4_ce                    # confusion matrix + curves
python scripts/make_gradcam.py outputs/effnet_b4_ce/best.pt    # Grad-CAM panel
python scripts/eval_external.py outputs/effnet_b4_ce/best.pt --dataset idrid  # domain generalisation
```

## 5. Interactive demo

```bash
streamlit run app/app.py
```

Upload a fundus image → preprocessing view + DR grade + confidence + referable-DR
decision + Grad-CAM overlay. Pick any trained checkpoint from the sidebar.

## Repository layout

```
src/
  config.py         central config + YAML loader
  preprocessing.py  auto-crop, Ben Graham, CLAHE
  dataset.py        APTOS Dataset + augmentation + dataloaders
  models.py         backbone factory (EfficientNet / ViT / Swin via timm)
  losses.py         Focal Loss, weighted CE
  metrics.py        QWK, per-class sens/spec, referable-DR summary
  engine.py         train / inference loops
scripts/
  download_data.py  Kaggle download helper
  prepare_splits.py stratified split + EDA
configs/            per-experiment YAML overrides
train.py            training entry point
```

## Contribution vs. libraries

`timm` (pretrained backbones), `albumentations`, OpenCV and PyTorch are used as
tools. The project's own contribution is the preprocessing design, the training/
evaluation pipeline, the controlled CNN-vs-ViT comparison, the domain-generalisation
study, and the analysis of results.
