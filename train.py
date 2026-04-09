from pathlib import Path
import re
import time
from typing import Dict, List
from datetime import datetime
import warnings
import shutil

warnings.filterwarnings("ignore", category=FutureWarning, module=r"denoising_diffusion_pytorch.*")
warnings.filterwarnings("ignore", message=r".*pin_memory.*no accelerator is found.*")

import matplotlib
import torch
import denoising_diffusion_pytorch.denoising_diffusion_pytorch as ddp_impl
import pandas as pd
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_DESC_RE = re.compile(r"loss:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


class TrainMetricsRecorder:
    """
    Records per-step metrics by intercepting tqdm loss updates from Trainer.train().
    """

    def __init__(self, output_csv: Path) -> None:
        self.output_csv = output_csv
        self.records: List[Dict[str, float]] = []
        self._orig_set_description = None
        self._start = 0.0
        self._last_t = 0.0
        self._last_step = 0

    def start(self) -> None:
        from tqdm import std as tqdm_std

        self._start = time.perf_counter()
        self._last_t = self._start
        self._orig_set_description = tqdm_std.tqdm.set_description

        def patched_set_description(pbar, desc=None, refresh=True):  # type: ignore[no-untyped-def]
            self._record_from_desc(pbar, desc)
            return self._orig_set_description(pbar, desc=desc, refresh=refresh)

        tqdm_std.tqdm.set_description = patched_set_description  # type: ignore[assignment]
        print(f"[INFO] Metrics recording enabled -> {self.output_csv.name}")

    def stop(self) -> None:
        from tqdm import std as tqdm_std

        if self._orig_set_description is not None:
            tqdm_std.tqdm.set_description = self._orig_set_description  # type: ignore[assignment]
        self._write_csv()

    def _record_from_desc(self, pbar, desc: str) -> None:  # type: ignore[no-untyped-def]
        if not desc:
            return
        m = LOSS_DESC_RE.search(str(desc))
        if not m:
            return
        try:
            loss = float(m.group(1))
        except Exception:
            return

        now = time.perf_counter()
        step = int(getattr(pbar, "n", 0))

        self.records.append(
            {
                "step": float(step),
                "loss": loss,
            }
        )
        self._last_t = now
        self._last_step = step

    def _write_csv(self) -> None:
        if not self.records:
            print("[WARN] No online metrics captured during training.")
            return
        try:
            import pandas as pd

            df = pd.DataFrame(self.records)
            df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step")
            df.to_csv(self.output_csv, index=False)
            print(f"[OK] Saved online metrics: {self.output_csv}")
        except Exception as exc:
            print(f"[WARN] Failed to save training_metrics.csv: {exc}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _overlay_label(img: Image.Image, label: str) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x, y = 8, 8
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return img


def _build_training_progress_gif(sample_files: List[Path], out_path: Path) -> bool:
    if not sample_files:
        return False
    stride = 1 if len(sample_files) <= 20 else (2 if len(sample_files) <= 40 else 3)
    frames = []
    for idx, p in enumerate(sample_files[::stride]):
        try:
            img = Image.open(p).convert("RGB")
            frames.append(_overlay_label(img, f"sample {idx + 1}"))
        except Exception:
            continue
    if not frames:
        return False
    try:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=800,
            loop=0,
        )
        _ok(f"Создан {out_path.name}")
        return True
    except Exception as exc:
        _warn(f"Не удалось создать GIF обучения: {exc}")
        return False


def _build_training_loss_curve(metrics_csv: Path, out_path: Path) -> bool:
    if not metrics_csv.exists():
        _warn("training_metrics.csv отсутствует, график loss пропущен.")
        return False
    try:
        df = pd.read_csv(metrics_csv)
        if "step" not in df.columns or "loss" not in df.columns:
            _warn("В training_metrics.csv нет нужных колонок step/loss.")
            return False
        d = df.dropna(subset=["step", "loss"]).copy()
        d["step"] = pd.to_numeric(d["step"], errors="coerce")
        d["loss"] = pd.to_numeric(d["loss"], errors="coerce")
        d = d.dropna(subset=["step", "loss"]).sort_values("step")
        if d.empty:
            _warn("training_metrics.csv пустой после фильтрации.")
            return False
        smooth_window = max(3, min(25, len(d) // 10 or 3))
        d["loss_smooth"] = d["loss"].rolling(window=smooth_window, min_periods=1).mean()
        plt.figure(figsize=(8, 5))
        plt.plot(d["step"], d["loss"], alpha=0.35, linewidth=1.0, label="loss")
        plt.plot(d["step"], d["loss_smooth"], linewidth=2.0, label=f"loss_smooth (w={smooth_window})")
        plt.title("Training Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        _ok(f"Создан {out_path.name}")
        return True
    except Exception as exc:
        _warn(f"Не удалось построить график loss: {exc}")
        return False


def finalize_training_layout(run_dir: Path, train_params: Dict[str, object], interrupted: bool) -> None:
    train_dir = run_dir / "train"
    model_samples_dir = train_dir / "model_samples"
    train_samples_dir = train_dir / "train_samples"
    model_samples_dir.mkdir(parents=True, exist_ok=True)
    train_samples_dir.mkdir(parents=True, exist_ok=True)

    sample_files = []
    for p in sorted(train_dir.glob("sample-*.png")):
        dst = train_samples_dir / p.name
        try:
            shutil.move(str(p), str(dst))
            sample_files.append(dst)
        except Exception as exc:
            _warn(f"Не удалось переместить {p.name}: {exc}")

    for p in sorted(train_dir.glob("model-*.pt")):
        try:
            shutil.move(str(p), str(model_samples_dir / p.name))
        except Exception as exc:
            _warn(f"Не удалось переместить {p.name}: {exc}")

    model_pt = train_dir / "model.pt"
    model_final = train_dir / "model-final.pt"
    if model_pt.exists():
        try:
            if model_final.exists():
                model_final.unlink()
            model_pt.rename(model_final)
            _ok("Переименован model.pt -> model-final.pt")
        except Exception as exc:
            _warn(f"Не удалось переименовать model.pt: {exc}")

    metrics_csv = train_dir / "training_metrics.csv"
    _build_training_loss_curve(metrics_csv, train_dir / "training_loss_curve.png")
    _build_training_progress_gif(sample_files, train_dir / "training_samples_progress.gif")

    final_loss = "N/A"
    total_steps = "N/A"
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if "step" in df.columns and not df["step"].dropna().empty:
                total_steps = str(int(float(df["step"].dropna().max())))
            if "loss" in df.columns and not df["loss"].dropna().empty:
                final_loss = f"{float(df['loss'].dropna().iloc[-1]):.6f}"
        except Exception:
            pass

    status_line = "Interrupted early by user." if interrupted else "Completed."
    report_text = f"""# Training Report

## Summary
- Status: {status_line}
- Dataset: {train_params.get("dataset", "MNIST")}
- Image size: {train_params.get("image_size", "32x32")}
- Steps reached: {total_steps}
- Final loss: {final_loss}

## Artifacts
- Model: `model-final.pt`
- Model checkpoints: `model_samples/`
- Training samples: `train_samples/`
- Metrics: `training_metrics.csv`
- Loss plot: `training_loss_curve.png`
- Samples GIF: `training_samples_progress.gif`

## Visuals
![Training Loss](training_loss_curve.png)
![Training Samples Progress](training_samples_progress.gif)
"""
    try:
        (train_dir / "training_report.md").write_text(report_text, encoding="utf-8")
        _ok("Создан training_report.md")
    except Exception as exc:
        _warn(f"Не удалось сохранить training_report.md: {exc}")


def prepare_mnist_image_folder(data_root: Path, output_root: Path, image_size: int = 32) -> None:
    """
    Downloads MNIST and exports it as PNG files so that Trainer can read an image folder.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    existing_png = list(output_root.glob("**/*.png"))
    if existing_png:
        print(f"Dataset already prepared: found {len(existing_png)} PNG files in {output_root}")
        return

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=preprocess,
    )

    print("Exporting MNIST images to PNG folder for Trainer...")
    for idx, (img_tensor, label) in enumerate(tqdm(dataset, total=len(dataset), desc="Preparing MNIST")):
        label_dir = output_root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = label_dir / f"mnist_{idx:05d}.png"
        to_pil_image(img_tensor).save(image_path)

    print(f"MNIST export finished. Saved {len(dataset)} images to {output_root}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    image_folder = data_root / "mnist_png" / "train"
    results_root = project_root / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"run_{run_id}"
    train_dir = run_dir / "train"
    inference_dir = run_dir / "inference"
    train_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    prepare_mnist_image_folder(data_root=data_root, output_root=image_folder, image_size=32)

    # Use single-process DataLoader to avoid noisy Windows spawn traceback on Ctrl+C.
    ddp_impl.cpu_count = lambda: 0

    model = Unet(
        dim=64,
        dim_mults=(1, 2),
        channels=1,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        folder=str(image_folder),
        train_batch_size=32,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=2000,
        save_and_sample_every=250,
        num_samples=25,
        results_folder=str(train_dir),
        amp=False,
        convert_image_to="L",
        calculate_fid=False,
    )

    print(f"[INFO] Starting training run: {run_id}")
    metrics_recorder = TrainMetricsRecorder(output_csv=train_dir / "training_metrics.csv")
    metrics_recorder.start()
    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARN] Training interrupted by user (Ctrl+C). Saving partial artifacts...")
    finally:
        metrics_recorder.stop()

    model_path = train_dir / "model.pt"
    torch.save(trainer.ema.ema_model.state_dict(), model_path)
    print(f"Saved EMA model weights to: {model_path}")

    train_params = {
        "dataset": "MNIST",
        "image_size": "32x32",
        "architecture": "DDPM + U-Net(dim=64, dim_mults=(1,2), channels=1)",
        "train_num_steps": 2000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_metrics_csv": "training_metrics.csv",
        "run_id": run_id,
    }
    finalize_training_layout(run_dir=run_dir, train_params=train_params, interrupted=interrupted)
    if interrupted:
        print(f"[OK] Partial report created in: {train_dir}")
    else:
        print(f"[OK] Training complete. Artifacts are in: {train_dir}")


if __name__ == "__main__":
    main()
