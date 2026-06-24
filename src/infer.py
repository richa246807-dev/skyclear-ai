"""Single entrypoint for Model 1, Model 2, SAR-ablation, and compositing inference."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models.baseline_inpaint import BaselineConfig, BaselineInpainter
from src.synthetic_clouds import mask_aware_composite


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration.

    Args:
        processed_dir: Root processed data directory.
        split: Split name to process.
        checkpoint: Model 1 checkpoint path.
        output_dir: Output directory for inference archives.
        baseline_backend: Baseline backend name.
        lama_command: Optional external LaMa command.
        stable_diffusion_model: Optional local Stable Diffusion inpainting model.
        allow_random_weights: Allow untrained Model 1 weights for smoke runs.
        allow_simple_baseline: Allow deterministic offline baseline fallback.
        base_channels: Model 1 width multiplier.
        device: Torch device string.
        limit: Optional maximum number of samples to process.
    """

    processed_dir: Path
    split: str
    checkpoint: Path
    output_dir: Path
    baseline_backend: str
    lama_command: str | None
    stable_diffusion_model: str | None
    allow_random_weights: bool
    allow_simple_baseline: bool
    base_channels: int
    device: str
    limit: int | None


def load_processed_sample(path: Path) -> dict[str, Any]:
    """Load a processed sample archive."""

    with np.load(path, allow_pickle=False) as sample:
        return {
            "cloudy_optical": sample["cloudy_optical"].astype(np.float32),
            "target_optical": sample["target_optical"].astype(np.float32),
            "sar": sample["sar"].astype(np.float32),
            "cloud_mask": sample["cloud_mask"].astype(np.float32),
            "metadata_json": str(sample["metadata_json"]),
        }


def build_model_input(
    cloudy_optical: np.ndarray,
    sar: np.ndarray,
    cloud_mask: np.ndarray,
    zero_sar: bool = False,
) -> np.ndarray:
    """Build the 7-channel Model 1 input array."""

    if cloudy_optical.ndim != 3 or sar.ndim != 3 or cloud_mask.ndim != 2:
        raise ValueError("Expected optical (C,H,W), SAR (2,H,W), and mask (H,W).")
    if cloudy_optical.shape[0] != 4:
        raise ValueError("Model 1 expects four optical input bands.")
    if sar.shape[0] != 2:
        raise ValueError("Model 1 expects two SAR input bands.")
    if cloudy_optical.shape[1:] != sar.shape[1:] or cloud_mask.shape != cloudy_optical.shape[1:]:
        raise ValueError("Optical, SAR, and mask spatial shapes must match.")
    sar_input = np.zeros_like(sar) if zero_sar else sar
    return np.concatenate([cloudy_optical, sar_input, cloud_mask[None, :, :]], axis=0).astype(np.float32)


def load_model1(
    checkpoint_path: Path,
    device: str,
    base_channels: int = 32,
    allow_random_weights: bool = False,
) -> Any:
    """Load the trained Model 1 generator."""

    import torch

    from src.models.generator import build_generator

    model = build_generator(base_channels=base_channels).to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["generator"] if "generator" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        LOGGER.info("Loaded Model 1 checkpoint %s", checkpoint_path)
    elif allow_random_weights:
        LOGGER.warning("Using randomly initialized Model 1 weights for a smoke run.")
    else:
        raise FileNotFoundError(f"Model 1 checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def predict_model1(
    model: Any,
    cloudy_optical: np.ndarray,
    sar: np.ndarray,
    cloud_mask: np.ndarray,
    device: str,
    zero_sar: bool = False,
) -> np.ndarray:
    """Run Model 1 for one tile."""

    import torch

    model_input = build_model_input(cloudy_optical, sar, cloud_mask, zero_sar=zero_sar)
    tensor = torch.from_numpy(model_input[None, :, :, :]).to(device)
    with torch.no_grad():
        prediction = model(tensor).detach().cpu().numpy()[0]
    return np.clip(prediction, 0.0, 1.0).astype(np.float32)


def run_sample_inference(
    sample: dict[str, Any],
    model1: Any,
    baseline: BaselineInpainter,
    device: str,
) -> dict[str, Any]:
    """Run both models and SAR ablation for one sample."""

    cloudy = sample["cloudy_optical"]
    target = sample["target_optical"]
    sar = sample["sar"]
    mask = sample["cloud_mask"]
    model1_raw = predict_model1(model1, cloudy, sar, mask, device=device, zero_sar=False)
    model1_zero_raw = predict_model1(model1, cloudy, sar, mask, device=device, zero_sar=True)
    model2_raw = baseline.inpaint(cloudy, mask)
    return {
        "cloudy_optical": cloudy,
        "target_optical": target,
        "sar": sar,
        "cloud_mask": mask,
        "model1_output": mask_aware_composite(cloudy, model1_raw, mask),
        "model1_sar_zero_output": mask_aware_composite(cloudy, model1_zero_raw, mask),
        "model2_output": mask_aware_composite(cloudy, model2_raw, mask),
        "metadata_json": sample["metadata_json"],
    }


def save_inference_output(path: Path, output: dict[str, Any]) -> None:
    """Save one inference output archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cloudy_optical=output["cloudy_optical"].astype(np.float32),
        target_optical=output["target_optical"].astype(np.float32),
        sar=output["sar"].astype(np.float32),
        cloud_mask=output["cloud_mask"].astype(np.float32),
        model1_output=output["model1_output"].astype(np.float32),
        model1_sar_zero_output=output["model1_sar_zero_output"].astype(np.float32),
        model2_output=output["model2_output"].astype(np.float32),
        metadata_json=output["metadata_json"],
    )


def run_inference(config: InferenceConfig) -> list[Path]:
    """Run inference on all samples in the configured split."""

    split_dir = config.processed_dir / config.split
    sample_paths = sorted(split_dir.glob("*.npz"))
    if config.limit is not None:
        sample_paths = sample_paths[: config.limit]
    if not sample_paths:
        raise FileNotFoundError(f"No samples found in {split_dir}.")

    model1 = load_model1(
        checkpoint_path=config.checkpoint,
        device=config.device,
        base_channels=config.base_channels,
        allow_random_weights=config.allow_random_weights,
    )
    baseline = BaselineInpainter(
        BaselineConfig(
            backend=config.baseline_backend,
            lama_command=config.lama_command,
            stable_diffusion_model=config.stable_diffusion_model,
            allow_simple_fallback=config.allow_simple_baseline,
        ),
        device=config.device,
    )

    saved: list[Path] = []
    for sample_path in sample_paths:
        sample = load_processed_sample(sample_path)
        output = run_sample_inference(sample, model1, baseline, device=config.device)
        metadata = json.loads(sample["metadata_json"])
        out_path = config.output_dir / f"{metadata['sample_id']}.npz"
        save_inference_output(out_path, output)
        saved.append(out_path)
        LOGGER.info("Saved inference output %s", out_path)
    return saved


def configure_logging() -> None:
    """Configure process logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run SkyClearAI inference.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model1_latest.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference/test"))
    parser.add_argument("--baseline-backend", default="auto", choices=["auto", "lama", "stable-diffusion", "sd", "simple"])
    parser.add_argument("--lama-command", default=None)
    parser.add_argument("--stable-diffusion-model", default=None)
    parser.add_argument("--allow-random-weights", action="store_true")
    parser.add_argument("--disable-simple-baseline", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the inference entrypoint."""

    configure_logging()
    args = parse_args()
    if args.device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config = InferenceConfig(
        processed_dir=args.processed_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        baseline_backend=args.baseline_backend,
        lama_command=args.lama_command,
        stable_diffusion_model=args.stable_diffusion_model,
        allow_random_weights=args.allow_random_weights,
        allow_simple_baseline=not args.disable_simple_baseline,
        base_channels=args.base_channels,
        device=device,
        limit=args.limit,
    )
    run_inference(config)


if __name__ == "__main__":
    main()

