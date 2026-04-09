from pathlib import Path
from datetime import datetime, timezone

import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image


def find_latest_run(results_root: Path) -> Path:
    run_dirs = [p for p in results_root.glob("run_*") if p.is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: p.name)
    for run_dir in reversed(run_dirs):
        if (run_dir / "train" / "model-final.pt").exists():
            return run_dir
    raise FileNotFoundError("Не найден запуск с model-final.pt. Сначала выполните train.py.")


def tensor_to_pil_gray(sample: torch.Tensor) -> Image.Image:
    sample_uint8 = ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return to_pil_image(sample_uint8)


def build_collage(images: list[Image.Image], out_path: Path, cols: int = 5, title: str | None = None) -> None:
    if not images:
        return
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    top_pad = 28 if title else 0
    canvas = Image.new("RGB", (cols * w, rows * h + top_pad), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img.convert("RGB"), (c * w, top_pad + r * h))
    canvas.save(out_path)


def build_generation_process_gif(trajectory: torch.Tensor, out_path: Path) -> None:
    # trajectory shape expected: [batch, timesteps, channels, h, w]
    if trajectory.ndim != 5 or trajectory.shape[0] == 0:
        return
    total_t = trajectory.shape[1]
    batch_size = trajectory.shape[0]
    stride = max(1, total_t // 40)
    frames = []

    cols = 5
    rows = (batch_size + cols - 1) // cols
    cell_w = trajectory.shape[-1]
    cell_h = trajectory.shape[-2]

    for i in range(0, total_t, stride):
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + 22), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for b in range(batch_size):
            r, c = divmod(b, cols)
            tile = tensor_to_pil_gray(trajectory[b, i]).convert("RGB")
            canvas.paste(tile, (c * cell_w, 22 + r * cell_h))
        label = f"denoise step {i}/{total_t - 1} (collage)"
        bbox = draw.textbbox((4, 4), label, font=font)
        draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255), font=font)
        frames.append(canvas)
    if not frames:
        return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], optimize=True, duration=120, loop=0)


def write_inference_report(
    infer_dir: Path,
    generated_files: list[Path],
    model_path: Path,
    generated_collage: Path | None = None,
    before_after_collage: Path | None = None,
    generation_gif: Path | None = None,
) -> None:
    report_path = infer_dir / "inference_report.md"
    created_at = datetime.now(timezone.utc).isoformat()
    lines = "\n".join([f"- `{p.name}`" for p in generated_files])
    visuals = []
    if generated_collage and generated_collage.exists():
        visuals.append("![Generated Collage](generated_collage.png)")
    if before_after_collage and before_after_collage.exists():
        visuals.append("![Before After Collage](before_after_collage.png)")
    if generation_gif and generation_gif.exists():
        visuals.append("![Generation Process GIF](generation_process.gif)")
    report_text = f"""# Inference Report

## Summary
- Time (UTC): {created_at}
- Model used: `{model_path}`
- Generated images: {len(generated_files)}

## Output
- Folder: `output/`
{lines if lines else "- No files generated."}

## Visuals
{chr(10).join(visuals) if visuals else "- No visuals generated."}
"""
    report_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_folder = project_root / "results"
    run_dir = find_latest_run(results_folder)
    inference_dir = run_dir / "inference"
    output_folder = inference_dir / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "train" / "model-final.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model file: {model_path}\n"
            "Run `python train.py` first."
        )

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

    state_dict = torch.load(model_path, map_location="cpu")
    diffusion.load_state_dict(state_dict)
    diffusion.eval()

    batch_size = 19
    with torch.no_grad():
        samples_all = diffusion.sample(batch_size=batch_size, return_all_timesteps=True).cpu()

    if samples_all.ndim == 5:
        final_samples = samples_all[:, -1]
        initial_noise = samples_all[:, 0]
    else:
        final_samples = samples_all
        initial_noise = final_samples

    for idx, sample in enumerate(final_samples):
        image = tensor_to_pil_gray(sample)
        image.save(output_folder / f"digit_{idx:02d}.png")

    # Collage of all generated outputs
    generated_files = sorted(output_folder.glob("digit_*.png"))
    generated_images = [Image.open(p).convert("RGB") for p in generated_files]
    generated_collage = inference_dir / "generated_collage.png"
    build_collage(generated_images, generated_collage, cols=5, title="Generated images (19)")

    # Before/after collage: noise vs generated
    noise_images = [tensor_to_pil_gray(x).convert("RGB") for x in initial_noise]
    noise_collage = inference_dir / "noise_collage.png"
    build_collage(noise_images, noise_collage, cols=5, title="Initial noise")

    before_after = inference_dir / "before_after_collage.png"
    if noise_collage.exists() and generated_collage.exists():
        left = Image.open(noise_collage).convert("RGB")
        right = Image.open(generated_collage).convert("RGB")
        h = max(left.height, right.height)
        canvas = Image.new("RGB", (left.width + right.width + 20, h), color=(255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 20, 0))
        canvas.save(before_after)

    # GIF of denoising process
    generation_gif = inference_dir / "generation_process.gif"
    if samples_all.ndim == 5:
        build_generation_process_gif(samples_all, generation_gif)

    write_inference_report(
        infer_dir=inference_dir,
        generated_files=generated_files,
        model_path=model_path,
        generated_collage=generated_collage,
        before_after_collage=before_after,
        generation_gif=generation_gif if generation_gif.exists() else None,
    )

    print(f"[OK] Saved {batch_size} generated images to: {output_folder}")


if __name__ == "__main__":
    main()
