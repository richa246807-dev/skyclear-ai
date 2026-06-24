"""Synthetic cloud masks, cloud transplanting, and compositing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CloudSynthesisConfig:
    """Configuration for synthetic cloud and shadow generation.

    Args:
        cloud_opacity_min: Minimum opacity for bright cloud overlay.
        cloud_opacity_max: Maximum opacity for bright cloud overlay.
        shadow_opacity_min: Minimum opacity for shifted cloud shadow.
        shadow_opacity_max: Maximum opacity for shifted cloud shadow.
        feather_radius: Approximate feather radius in pixels.
    """

    cloud_opacity_min: float = 0.45
    cloud_opacity_max: float = 0.9
    shadow_opacity_min: float = 0.12
    shadow_opacity_max: float = 0.35
    feather_radius: int = 4


def _rng(seed: int | None) -> np.random.Generator:
    """Return a NumPy random generator."""

    return np.random.default_rng(seed)


def _blur_3x3(image: Array, passes: int) -> Array:
    """Blur an image using repeated 3x3 mean filters.

    Args:
        image: Two-dimensional image.
        passes: Number of blur passes.

    Returns:
        Blurred image with the same shape as ``image``.
    """

    output = image.astype(np.float32, copy=True)
    for _ in range(max(0, passes)):
        padded = np.pad(output, 1, mode="edge")
        output = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return output


def feather_mask(mask: Array, radius: int = 4) -> Array:
    """Feather a binary or soft mask into the closed range [0, 1].

    Args:
        mask: Two-dimensional binary or soft mask.
        radius: Approximate feather radius in pixels.

    Returns:
        Soft mask with cloud centers near 1 and boundaries in (0, 1).

    Raises:
        ValueError: If ``mask`` is not two-dimensional.
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be two-dimensional.")
    clipped = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if radius <= 0:
        return clipped
    blurred = _blur_3x3(clipped, passes=radius)
    return np.clip(np.maximum(clipped, blurred), 0.0, 1.0).astype(np.float32)


def generate_realistic_cloud_mask(
    height: int,
    width: int,
    seed: int | None = None,
    coverage: tuple[float, float] = (0.12, 0.45),
    feather_radius: int = 4,
) -> Array:
    """Generate a soft, irregular cloud mask.

    Args:
        height: Output mask height in pixels.
        width: Output mask width in pixels.
        seed: Optional deterministic seed.
        coverage: Inclusive target cloud coverage range.
        feather_radius: Approximate feather radius in pixels.

    Returns:
        Soft cloud mask with shape ``(height, width)`` and values in [0, 1].

    Raises:
        ValueError: If dimensions or coverage are invalid.
    """

    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive.")
    min_cov, max_cov = coverage
    if not 0.0 < min_cov <= max_cov < 1.0:
        raise ValueError("Coverage must satisfy 0 < min <= max < 1.")

    rng = _rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.zeros((height, width), dtype=np.float32)
    n_blobs = int(rng.integers(8, 22))
    for _ in range(n_blobs):
        cy = float(rng.uniform(0, height))
        cx = float(rng.uniform(0, width))
        sy = float(rng.uniform(height * 0.04, height * 0.18))
        sx = float(rng.uniform(width * 0.04, width * 0.2))
        amplitude = float(rng.uniform(0.6, 1.4))
        blob = np.exp(-(((yy - cy) ** 2) / (2.0 * sy**2) + ((xx - cx) ** 2) / (2.0 * sx**2)))
        field += amplitude * blob.astype(np.float32)

    field += 0.25 * rng.random((height, width), dtype=np.float32)
    field = _blur_3x3(field, passes=3)
    target_coverage = float(rng.uniform(min_cov, max_cov))
    threshold = float(np.quantile(field, 1.0 - target_coverage))
    binary = (field >= threshold).astype(np.float32)
    return feather_mask(binary, radius=feather_radius)


def make_synthetic_clear_tile(
    height: int,
    width: int,
    bands: int = 4,
    seed: int | None = None,
) -> Array:
    """Create a deterministic synthetic optical tile in [0, 1].

    Args:
        height: Tile height.
        width: Tile width.
        bands: Number of optical bands.
        seed: Optional deterministic seed.

    Returns:
        Array with shape ``(bands, height, width)``.
    """

    if bands <= 0:
        raise ValueError("Band count must be positive.")
    rng = _rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    yy = yy / max(1, height - 1)
    xx = xx / max(1, width - 1)
    base = np.empty((bands, height, width), dtype=np.float32)
    for band in range(bands):
        phase = float(rng.uniform(0.0, np.pi))
        texture = 0.08 * np.sin((band + 1) * np.pi * xx + phase)
        texture += 0.06 * np.cos((band + 2) * np.pi * yy - phase)
        base[band] = 0.25 + 0.45 * xx + 0.2 * yy + texture
        base[band] += 0.03 * rng.normal(size=(height, width)).astype(np.float32)
    if bands >= 4:
        base[3] = np.maximum(base[3], base[2] + 0.08)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def make_synthetic_sar_tile(
    optical: Array,
    seed: int | None = None,
) -> Array:
    """Create deterministic VV/VH-like SAR channels from an optical tile.

    Args:
        optical: Optical tile with shape ``(bands, height, width)``.
        seed: Optional deterministic seed.

    Returns:
        SAR tile with shape ``(2, height, width)`` and values in [0, 1].
    """

    if optical.ndim != 3:
        raise ValueError("Optical tile must have shape (bands, height, width).")
    rng = _rng(seed)
    gray = np.mean(optical[: min(3, optical.shape[0])], axis=0)
    vv = 0.55 * gray + 0.25 * _blur_3x3(gray, passes=2)
    vh = 0.35 * gray + 0.45 * np.abs(np.gradient(gray)[0])
    sar = np.stack([vv, vh], axis=0)
    sar += 0.025 * rng.normal(size=sar.shape).astype(np.float32)
    return np.clip(sar, 0.0, 1.0).astype(np.float32)


def transplant_clouds(
    clear_optical: Array,
    cloud_mask: Array,
    seed: int | None = None,
    config: CloudSynthesisConfig | None = None,
) -> Array:
    """Transplant a cloud mask onto a clear optical tile.

    Args:
        clear_optical: Clear optical tile with shape ``(bands, height, width)``.
        cloud_mask: Soft cloud mask with shape ``(height, width)``.
        seed: Optional deterministic seed.
        config: Optional synthesis parameters.

    Returns:
        Cloudy optical tile with shape matching ``clear_optical``.

    Raises:
        ValueError: If shapes are incompatible.
    """

    if clear_optical.ndim != 3:
        raise ValueError("Clear optical tile must have shape (bands, height, width).")
    bands, height, width = clear_optical.shape
    if cloud_mask.shape != (height, width):
        raise ValueError("Cloud mask shape must match optical height and width.")

    cfg = config or CloudSynthesisConfig()
    rng = _rng(seed)
    mask = feather_mask(cloud_mask, radius=cfg.feather_radius)
    opacity = float(rng.uniform(cfg.cloud_opacity_min, cfg.cloud_opacity_max))
    cloud_color = rng.uniform(0.78, 1.0, size=(bands, 1, 1)).astype(np.float32)
    cloudy = clear_optical * (1.0 - opacity * mask[None, :, :])
    cloudy += cloud_color * opacity * mask[None, :, :]

    shift_y = int(rng.integers(max(1, height // 40), max(2, height // 12)))
    shift_x = int(rng.integers(max(1, width // 40), max(2, width // 12)))
    shadow = np.zeros_like(mask)
    shadow[shift_y:, shift_x:] = mask[: height - shift_y, : width - shift_x]
    shadow = feather_mask(shadow, radius=cfg.feather_radius)
    shadow_opacity = float(rng.uniform(cfg.shadow_opacity_min, cfg.shadow_opacity_max))
    visible_shadow = np.clip(shadow - mask, 0.0, 1.0)
    cloudy *= 1.0 - shadow_opacity * visible_shadow[None, :, :]

    return np.clip(cloudy, 0.0, 1.0).astype(np.float32)


def estimate_cloud_mask_from_optical(optical: Array, feather_radius: int = 4) -> Array:
    """Estimate a cloud mask from optical bands for demo-time GeoTIFF input.

    Args:
        optical: Optical tile with shape ``(bands, height, width)`` in [0, 1].
        feather_radius: Approximate feather radius in pixels.

    Returns:
        Soft cloud mask with shape ``(height, width)``.
    """

    if optical.ndim != 3:
        raise ValueError("Optical tile must have shape (bands, height, width).")
    from src.constants import BAND_RED, BAND_NIR
    visible = optical[: min(3, optical.shape[0])]
    brightness = np.mean(visible, axis=0)
    if optical.shape[0] >= 4:
        red = optical[BAND_RED]
        nir = optical[BAND_NIR]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        raw = (brightness > 0.62) & (ndvi < 0.35)
    else:
        raw = brightness > 0.68
    if not np.any(raw):
        threshold = float(np.quantile(brightness, 0.88))
        raw = brightness >= threshold
    return feather_mask(raw.astype(np.float32), radius=feather_radius)


def mask_aware_composite(
    original: Array,
    generated: Array,
    cloud_mask: Array,
    feather_radius: int = 4,
) -> Array:
    """Composite generated pixels inside the cloud mask and preserve the original elsewhere.

    Args:
        original: Original cloudy optical tile with shape ``(bands, height, width)``.
        generated: Generated optical tile with shape matching ``original``.
        cloud_mask: Cloud mask with shape ``(height, width)``.
        feather_radius: Approximate boundary feather radius in pixels.

    Returns:
        Composite tile with shape matching ``original``.

    Raises:
        ValueError: If input shapes are incompatible.
    """

    if original.shape != generated.shape:
        raise ValueError("Original and generated tiles must have the same shape.")
    if original.ndim != 3:
        raise ValueError("Tiles must have shape (bands, height, width).")
    if cloud_mask.shape != original.shape[1:]:
        raise ValueError("Cloud mask shape must match tile height and width.")
    alpha = feather_mask(cloud_mask, radius=feather_radius)[None, :, :]
    composite = original * (1.0 - alpha) + generated * alpha
    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def get_s2cloudless_mask(bands_10_data: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Compute a cloud mask using s2cloudless from a 10-band Sentinel-2 image.

    Expected band order: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12.
    """
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError as exc:
        raise ImportError("s2cloudless is required for real cloud mask extraction. Install it with pip.") from exc

    data_hwc = np.transpose(bands_10_data, (1, 2, 0)).astype(np.float32)
    if np.max(data_hwc) > 2.0:
        data_hwc = data_hwc / 10000.0
    data_hwc = np.clip(data_hwc, 0.0, 1.0)
    detector = S2PixelCloudDetector(threshold=threshold, average_over=4, dilation_size=2)
    mask = detector.get_cloud_mask(data_hwc)
    return mask.astype(np.float32)

