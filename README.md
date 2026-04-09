# mnist-ddpm-research

DDPM pipeline for MNIST with isolated run artifacts and concise markdown reports for training and inference.

## Key points

- Training and generation are split into two scripts: `train.py` and `generate.py`.
- Each training run is isolated in `results/run_<timestamp>/`.
- Reports are generated automatically:
  - `results/run_<timestamp>/train/training_report.md`
  - `results/run_<timestamp>/inference/inference_report.md`
- Generation includes visual artifacts: output grid collage, before/after collage, and denoising GIF.

## Dataset policy

- The repository does **not** store the MNIST dataset.
- On first run, `train.py` downloads MNIST and prepares image folders under `data/`.
- This is intentional and keeps the repository lightweight.
- Dataset artifacts are ignored by git (`data/MNIST/`, `data/mnist_png/`).

## Project structure

```text
ddpm_project/
├── data/                         # local dataset cache (not committed)
├── results/                      # run-scoped artifacts
│   └── run_YYYYMMDD_HHMMSS/
│       ├── train/
│       │   ├── model-final.pt
│       │   ├── model_samples/
│       │   ├── train_samples/
│       │   ├── training_metrics.csv
│       │   ├── training_loss_curve.png
│       │   ├── training_samples_progress.gif
│       │   └── training_report.md
│       └── inference/
│           ├── output/
│           ├── generated_collage.png
│           ├── noise_collage.png
│           ├── before_after_collage.png
│           ├── generation_process.gif
│           └── inference_report.md
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

## Quick start

### 1) Create environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Train

```bash
python train.py
```

### 3) Generate from latest run

```bash
python generate.py
```

## Notes

- Model: U-Net (`dim=64`, `dim_mults=(1,2)`, `channels=1`)
- Diffusion timesteps: `1000`
- Default training steps: `2000`
- CPU-first configuration (`amp=False`)
