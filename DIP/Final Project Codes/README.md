# Noise Removal in Medical Images
### Digital Image Processing — Final Project

**Student:** Mardonali Fayzullayev  
**ID:** 230423

---

## Project Structure

```
medical_denoising/
│
├── main.py                  ← Run everything from here
├── config.py                ← All settings (paths, hyperparameters)
├── requirements.txt         ← pip dependencies
│
├── src/
│   ├── dataset.py           ← Load data, add noise
│   ├── filters.py           ← Median, Gaussian, Bilateral, NLM
│   ├── dncnn.py             ← DnCNN deep learning model
│   ├── metrics.py           ← PSNR, SSIM, Edge similarity
│   └── visualize.py         ← All plots and charts
│
├── data/
│   ├── raw/                 ← Put your Kaggle zip here
│   ├── clean/               ← Preprocessed clean images (.npy)
│   └── noisy/               ← Noisy image variants (.npy)
│
├── results/
│   ├── classical/
│   ├── nlm/
│   ├── dncnn/
│   └── metrics/             ← CSV results tables
│
├── models/                  ← Saved DnCNN weights (.pth)
└── plots/                   ← All generated figures (.png)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Kaggle API (automatic):**
```bash
# 1. Download kaggle.json from: kaggle.com → Profile → Settings → API
# 2. Run:
python main.py --step 1 --kaggle path/to/kaggle.json
```

**Option B — Manual download:**
```
1. Go to: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
2. Click Download → save archive.zip
3. Run:
```
```bash
python main.py --step 1 --zip path/to/archive.zip
```

**Option C — No download (demo mode):**
```bash
# Uses scikit-image built-in brain MRI images
python main.py --step 1
```

---

## Run the Full Pipeline

```bash
# Run everything at once
python main.py --all

# Or run step by step:
python main.py --step 1    # Dataset & preprocessing
python main.py --step 2    # Classical filters
python main.py --step 3    # DnCNN training
python main.py --step 4    # Evaluation & plots
```

---

## Methods Implemented

| # | Method | Type | Best For |
|---|--------|------|----------|
| 1 | Median Filter | Classical | Salt & Pepper noise |
| 2 | Gaussian Filter | Classical | Gaussian noise |
| 3 | Bilateral Filter | Advanced Classical | Gaussian + edge preservation |
| 4 | Non-Local Means (NLM) | Advanced Classical | Any noise, best detail |
| 5 | DnCNN | Deep Learning | State-of-the-art denoising |

---

## Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| PSNR | Peak Signal-to-Noise Ratio | > 35 dB |
| SSIM | Structural Similarity Index | > 0.90 |
| Edge Sim | Canny edge overlap (F1) | Closer to 1.0 |

---

## Output Files

After running the pipeline you will find:

- `plots/noise_samples.png` — noise types visualization
- `plots/filter_comparison_gaussian_20.png` — side-by-side results
- `plots/edge_maps_comparison.png` — edge preservation analysis
- `plots/psnr_bar_*.png` — PSNR bar charts
- `plots/ssim_bar_*.png` — SSIM bar charts
- `plots/dncnn_training_curve.png` — training loss
- `plots/summary_dashboard.png` — full results dashboard
- `results/metrics/full_results.csv` — complete numerical results

---

## References

1. Gonzalez & Woods, *Digital Image Processing*, 4th ed., Pearson, 2018  
2. Tomasi & Manduchi, "Bilateral filtering", ICCV 1998  
3. Buades et al., "Non-local means denoising", CVPR 2005  
4. Zhang et al., "DnCNN", IEEE TIP 2017  
