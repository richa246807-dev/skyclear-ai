"""Zero-shot inpainting baseline adapters for Model 2."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the zero-shot baseline adapter.

    Args:
        backend: Baseline backend. ``auto`` prefers LaMa, then Stable Diffusion, then simple.
        lama_command: Optional external LaMa command.
        stable_diffusion_model: Optional local Stable Diffusion inpainting model path or id.
        allow_simple_fallback: Whether to allow deterministic offline fallback for smoke tests.
    """

    backend: str = "auto"
    lama_command: str | None = None
    stable_diffusion_model: str | None = None
    allow_simple_fallback: bool = True


def _optical_to_rgb_uint8(optical: np.ndarray) -> np.ndarray:
    """Convert channel-first optical data to RGB uint8."""

    from src.constants import BAND_RED, BAND_GREEN, BAND_BLUE
    if optical.ndim != 3:
        raise ValueError("Optical input must have shape (bands, height, width).")
    if optical.shape[0] >= 4:
        rgb = np.stack([optical[BAND_RED], optical[BAND_GREEN], optical[BAND_BLUE]], axis=-1)
    elif optical.shape[0] >= 3:
        rgb = np.moveaxis(optical[:3], 0, -1)
    else:
        rgb = np.repeat(optical[0, :, :, None], 3, axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _rgb_uint8_to_optical(rgb: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Project RGB inpainting output back to the original optical band layout."""

    from src.constants import BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR
    rgb_float = rgb.astype(np.float32) / 255.0
    output = original.copy()
    if original.shape[0] >= 4:
        output[BAND_RED] = rgb_float[:, :, 0]
        output[BAND_GREEN] = rgb_float[:, :, 1]
        output[BAND_BLUE] = rgb_float[:, :, 2]
        output[BAND_NIR] = np.clip(original[BAND_NIR] + np.mean(output[:3] - original[:3], axis=0), 0.0, 1.0)
    elif original.shape[0] >= 3:
        output[:3] = np.moveaxis(rgb_float, -1, 0)
    else:
        output[0] = np.mean(rgb_float, axis=-1)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _write_lama_inputs(input_dir: Path, optical: np.ndarray, cloud_mask: np.ndarray) -> tuple[Path, Path]:
    """Write RGB image and binary mask files for an external LaMa command."""

    image_path = input_dir / "input.png"
    mask_path = input_dir / "input_mask.png"
    Image.fromarray(_optical_to_rgb_uint8(optical)).save(image_path)
    mask_uint8 = (np.clip(cloud_mask, 0.0, 1.0) > 0.2).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(mask_path)
    return image_path, mask_path


def _run_external_lama(command: str, optical: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Run an external LaMa command and return optical output.

    The command receives three appended arguments: input PNG path, mask PNG path, and output
    PNG path. This works with small wrapper scripts around common LaMa checkouts while keeping
    this repository independent of a specific checkout layout.
    """

    with tempfile.TemporaryDirectory(prefix="skyclear_lama_") as tmp:
        tmp_path = Path(tmp)
        image_path, mask_path = _write_lama_inputs(tmp_path, optical, cloud_mask)
        output_path = tmp_path / "output.png"
        args = command.split() + [str(image_path), str(mask_path), str(output_path)]
        LOGGER.info("Running LaMa command: %s", " ".join(args[:1]))
        subprocess.run(args, check=True, capture_output=True, text=True)
        if not output_path.exists():
            raise RuntimeError("LaMa command finished without writing the expected output image.")
        rgb = np.asarray(Image.open(output_path).convert("RGB"))
        return _rgb_uint8_to_optical(rgb, optical)


def _run_stable_diffusion(
    model_name_or_path: str,
    optical: np.ndarray,
    cloud_mask: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run a local Stable Diffusion inpainting pipeline."""

    try:
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise RuntimeError("diffusers is required for the Stable Diffusion fallback.") from exc

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name_or_path)
    pipe = pipe.to(device)
    image = Image.fromarray(_optical_to_rgb_uint8(optical)).convert("RGB")
    mask = Image.fromarray((np.clip(cloud_mask, 0.0, 1.0) > 0.2).astype(np.uint8) * 255)
    result = pipe(
        prompt="satellite image surface reconstruction",
        image=image,
        mask_image=mask,
        num_inference_steps=20,
        guidance_scale=1.0,
    ).images[0]
    return _rgb_uint8_to_optical(np.asarray(result.convert("RGB")), optical)


def _simple_offline_inpaint(optical: np.ndarray, cloud_mask: np.ndarray, iterations: int = 80) -> np.ndarray:
    """Fill masked pixels by iterative neighbor averaging for offline smoke runs."""

    if optical.ndim != 3 or cloud_mask.ndim != 2:
        raise ValueError("Expected optical shape (bands, height, width) and mask shape (height, width).")
    if cloud_mask.shape != optical.shape[1:]:
        raise ValueError("Cloud mask shape must match optical height and width.")
    hard_mask = cloud_mask > 0.2
    output = optical.copy().astype(np.float32)
    if not np.any(hard_mask):
        return output
    known = ~hard_mask
    for band in range(output.shape[0]):
        fill_value = float(np.mean(output[band][known])) if np.any(known) else 0.5
        output[band][hard_mask] = fill_value
    for _ in range(iterations):
        padded = np.pad(output, ((0, 0), (1, 1), (1, 1)), mode="edge")
        averaged = (
            padded[:, :-2, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2]
            + padded[:, 1:-1, 2:]
        ) / 4.0
        output[:, hard_mask] = averaged[:, hard_mask]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


class BaselineInpainter:
    """Model 2 zero-shot baseline adapter."""

    def __init__(self, config: BaselineConfig | None = None, device: str = "cpu") -> None:
        """Initialize the adapter.

        Args:
            config: Backend configuration.
            device: Torch device string used by Stable Diffusion if selected.
        """

        self.config = config or BaselineConfig()
        self.device = device

    def inpaint(self, optical: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """Run the selected inpainting baseline.

        Args:
            optical: Cloudy optical tile with shape ``(bands, height, width)``.
            cloud_mask: Cloud mask with shape ``(height, width)``.

        Returns:
            Inpainted optical tile with shape matching ``optical``.
        """

        backend = self.config.backend.lower()
        if backend == "auto":
            if self.config.lama_command or os.getenv("SKYCLEAR_LAMA_COMMAND"):
                backend = "lama"
            elif self.config.stable_diffusion_model or os.getenv("SKYCLEAR_SD_INPAINT_MODEL"):
                backend = "stable-diffusion"
            else:
                backend = "simple"

        if backend == "lama":
            command = self.config.lama_command or os.getenv("SKYCLEAR_LAMA_COMMAND")
            if command is None or shutil.which(command.split()[0]) is None:
                if not self.config.allow_simple_fallback:
                    raise RuntimeError("LaMa command is not configured or not executable.")
                LOGGER.warning("LaMa is unavailable; using offline inpainting fallback.")
                return _simple_offline_inpaint(optical, cloud_mask)
            return _run_external_lama(command, optical, cloud_mask)

        if backend in {"stable-diffusion", "sd"}:
            model_name = self.config.stable_diffusion_model or os.getenv("SKYCLEAR_SD_INPAINT_MODEL")
            if not model_name:
                if not self.config.allow_simple_fallback:
                    raise RuntimeError("Stable Diffusion inpainting model path is not configured.")
                LOGGER.warning("Stable Diffusion is unavailable; using offline inpainting fallback.")
                return _simple_offline_inpaint(optical, cloud_mask)
            return _run_stable_diffusion(model_name, optical, cloud_mask, self.device)

        if backend == "simple":
            if not self.config.allow_simple_fallback:
                raise RuntimeError("Simple fallback disabled and no pretrained baseline is configured.")
            return _simple_offline_inpaint(optical, cloud_mask)

        raise ValueError(f"Unsupported baseline backend: {self.config.backend}")

