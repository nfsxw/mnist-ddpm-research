# DDPM MNIST (CPU Demo)

This project trains a DDPM model on MNIST digits (`0-9`) using `denoising_diffusion_pytorch`.
The setup is intentionally lightweight for CPU-only machines.

## Project structure

```text
ddpm_project/
├── data/           # MNIST download + exported PNG dataset
├── results/        # per-run folders: run_YYYYMMDD_HHMMSS with isolated artifacts
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

## 1) Create environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you need to force CPU-only PyTorch wheels, use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install denoising-diffusion-pytorch tqdm
```

## 3) Train model

```bash
python train.py
```

Training details:
- MNIST images are resized to `32x32`
- Input normalization is handled as `[-1, 1]` by the trainer pipeline
- U-Net config: `dim=64`, `dim_mults=(1, 2)`, `channels=1`
- Diffusion timesteps: `1000`
- AMP is disabled (`amp=False`) for CPU compatibility
- Demo training steps: `2000`
- Intermediate samples/checkpoints are saved every `500` steps in `results/` (`num_samples=16` for 4x4 grid)
- Every run is isolated in `results/run_<timestamp>/train/`
- Final model is saved as `model-final.pt`
- Checkpoint models are moved to `model_samples/`
- Training sample images are moved to `train_samples/`
- Automatic training report: `training_report.md`
- Loss graph: `training_loss_curve.png`

## 4) Generate images from latest run

```bash
python generate.py
```

This creates `19` images in:
- `results/run_<latest>/inference/output/`

Also creates:
- `results/run_<latest>/inference/inference_report.md`
- `results/run_<latest>/inference/generated_collage.png` (all generated images)
- `results/run_<latest>/inference/before_after_collage.png` (noise collage vs generated collage)
- `results/run_<latest>/inference/generation_process.gif` (denoising process)

## Note on runtime

CPU training is slower than GPU training. On a typical consumer CPU, this demo can take around **1-2 hours** (sometimes more depending on your hardware and background load).
