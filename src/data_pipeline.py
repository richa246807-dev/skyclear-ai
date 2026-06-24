"""Single entrypoint for building SkyClearAI training, validation, and test tiles."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.synthetic_clouds import (
    estimate_cloud_mask_from_optical,
    generate_realistic_cloud_mask,
    make_synthetic_clear_tile,
    make_synthetic_sar_tile,
    transplant_clouds,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleMetadata:
    """Metadata stored with each processed tile.

    Args:
        sample_id: Stable sample identifier.
        split: Dataset split name.
        source: Source path or synthetic source label.
        cloud_fraction: Mean hard cloud fraction.
        patch_size: Tile size in pixels.
    """

    sample_id: str
    split: str
    source: str
    cloud_fraction: float
    patch_size: int


def configure_logging() -> None:
    """Configure process logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _normalize_raster(array: np.ndarray) -> np.ndarray:
    """Normalize a raster array into [0, 1]."""

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("Raster contains no finite pixels.")
    valid = array[finite]
    if valid.max() <= 1.5 and valid.min() >= -0.1:
        return np.clip(array, 0.0, 1.0).astype(np.float32)
    high = float(np.percentile(valid, 99.5))
    if high <= 0.0:
        raise ValueError("Raster normalization scale must be positive.")
    return np.clip(array / high, 0.0, 1.0).astype(np.float32)


def _read_geotiff(path: Path, max_bands: int | None = None) -> np.ndarray:
    """Read a GeoTIFF into a channel-first float array."""

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to read GeoTIFF inputs.") from exc

    with rasterio.open(path) as dataset:
        count = dataset.count if max_bands is None else min(dataset.count, max_bands)
        array = dataset.read(list(range(1, count + 1)))
    return _normalize_raster(array)


def _iter_tiles(array: np.ndarray, patch_size: int) -> Iterable[tuple[int, int, np.ndarray]]:
    """Yield non-overlapping channel-first tiles from an array."""

    if array.ndim != 3:
        raise ValueError("Input array must have shape (bands, height, width).")
    _, height, width = array.shape
    for row in range(0, height - patch_size + 1, patch_size):
        for col in range(0, width - patch_size + 1, patch_size):
            yield row, col, array[:, row : row + patch_size, col : col + patch_size]


def _split_for_index(index: int, total: int, train_fraction: float, val_fraction: float) -> str:
    """Return a deterministic split name for a sample index."""

    train_cutoff = int(total * train_fraction)
    val_cutoff = train_cutoff + int(total * val_fraction)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def save_processed_sample(
    output_dir: Path,
    metadata: SampleMetadata,
    cloudy_optical: np.ndarray,
    target_optical: np.ndarray,
    sar: np.ndarray,
    cloud_mask: np.ndarray,
) -> Path:
    """Save one processed sample as a compressed NumPy archive.

    Args:
        output_dir: Root processed data directory.
        metadata: Sample metadata.
        cloudy_optical: Cloudy optical tile with shape ``(4, H, W)``.
        target_optical: Clear target tile with shape ``(4, H, W)``.
        sar: SAR tile with shape ``(2, H, W)``.
        cloud_mask: Soft cloud mask with shape ``(H, W)``.

    Returns:
        Path to the saved ``.npz`` file.
    """

    split_dir = output_dir / metadata.split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{metadata.sample_id}.npz"
    np.savez_compressed(
        path,
        cloudy_optical=cloudy_optical.astype(np.float32),
        target_optical=target_optical.astype(np.float32),
        sar=sar.astype(np.float32),
        cloud_mask=cloud_mask.astype(np.float32),
        metadata_json=json.dumps(asdict(metadata), sort_keys=True),
    )
    return path


def build_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 24,
    patch_size: int = 256,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> list[Path]:
    """Build a synthetic Sentinel-2-like processed dataset.

    Args:
        output_dir: Root processed data directory.
        num_samples: Number of samples to create.
        patch_size: Patch size in pixels.
        seed: Deterministic seed.
        train_fraction: Fraction assigned to train split.
        val_fraction: Fraction assigned to validation split.

    Returns:
        Saved sample paths.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("Split fractions must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave a test split.")

    saved: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_samples):
        split = _split_for_index(index, num_samples, train_fraction, val_fraction)
        clear = make_synthetic_clear_tile(patch_size, patch_size, bands=4, seed=seed + index)
        mask = generate_realistic_cloud_mask(patch_size, patch_size, seed=seed + 10_000 + index)
        cloudy = transplant_clouds(clear, mask, seed=seed + 20_000 + index)
        sar = make_synthetic_sar_tile(clear, seed=seed + 30_000 + index)
        metadata = SampleMetadata(
            sample_id=f"synthetic_{index:05d}",
            split=split,
            source="synthetic_sentinel2_fixture",
            cloud_fraction=float(np.mean(mask > 0.5)),
            patch_size=patch_size,
        )
        saved.append(save_processed_sample(output_dir, metadata, cloudy, clear, sar, mask))
    LOGGER.info("Saved %d synthetic samples under %s", len(saved), output_dir)
    return saved


def build_from_geotiffs(
    clear_dir: Path,
    output_dir: Path,
    cloud_probability_dir: Path | None,
    patch_size: int = 256,
    seed: int = 42,
) -> list[Path]:
    """Build processed samples from local Sentinel-2 and Sentinel-1 GeoTIFF scenes."""

    clear_path = clear_dir / "sentinel2_clear.tif"
    cloudy_path = clear_dir / "sentinel2_cloudy.tif"
    sar_path = clear_dir / "sentinel1_grd.tif"

    if not clear_path.exists():
        raise FileNotFoundError(f"Missing {clear_path}")

    # Read the full aligned rasters
    clear_full = _read_geotiff(clear_path)
    if clear_full.shape[0] < 4:
        raise ValueError(f"{clear_path} must contain at least four optical bands.")

    has_real_cloudy = cloudy_path.exists()
    has_real_sar = sar_path.exists()

    if has_real_cloudy:
        from s2cloudless import S2PixelCloudDetector
        cloudy_full = _read_geotiff(cloudy_path)
        # S2Cloudless expects 10 bands: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
        # Our cloudy image has 11 bands: B01, B02, B03, B04, B05, B08, B8A, B09, B10, B11, B12
        # So we skip B03 (index 2)
        s2c_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
        s2c_input = cloudy_full[s2c_indices]
        
        cloud_detector = S2PixelCloudDetector(
            threshold=0.4, average_over=4, dilation_size=2, all_bands=False
        )
        # s2cloudless expects shape (1, H, W, C) in range [0, 1]
        s2c_input_hwc = np.transpose(s2c_input, (1, 2, 0))[np.newaxis, ...]
        mask_full = cloud_detector.get_cloud_probability_maps(s2c_input_hwc)[0].astype(np.float32)
        
        # Extract optical bands for the model: B02, B03, B04, B08 (indices 1, 2, 3, 5)
        cloudy_optical_full = cloudy_full[[1, 2, 3, 5]]
    else:
        mask_full = None
        cloudy_optical_full = None

    if has_real_sar:
        sar_full = _read_geotiff(sar_path)
    else:
        sar_full = None

    saved: list[Path] = []
    rng = np.random.default_rng(seed)
    
    all_tiles: list[tuple[int, int, np.ndarray]] = list(_iter_tiles(clear_full, patch_size))
    total = len(all_tiles)
    if total == 0:
        raise ValueError("No full-size tiles could be extracted from the GeoTIFF scenes.")

    for index, (row, col, clear) in enumerate(all_tiles):
        split = _split_for_index(index, total, train_fraction=0.7, val_fraction=0.15)
        
        if has_real_cloudy:
            mask = mask_full[row : row + patch_size, col : col + patch_size]
            cloudy = cloudy_optical_full[:, row : row + patch_size, col : col + patch_size]
        else:
            mask = estimate_cloud_mask_from_optical(clear)
            cloudy = transplant_clouds(clear, mask, seed=int(rng.integers(0, 2**31 - 1)))
            
        if has_real_sar:
            sar = sar_full[:, row : row + patch_size, col : col + patch_size]
        else:
            sar = make_synthetic_sar_tile(clear, seed=int(rng.integers(0, 2**31 - 1)))
            
        sample_id = f"{clear_path.stem}_{row:05d}_{col:05d}"
        metadata = SampleMetadata(
            sample_id=sample_id,
            split=split,
            source=str(clear_path),
            cloud_fraction=float(np.mean(mask > 0.5)),
            patch_size=patch_size,
        )
        saved.append(save_processed_sample(output_dir, metadata, cloudy, clear, sar, mask))
        
    LOGGER.info("Saved %d GeoTIFF-derived samples under %s", len(saved), output_dir)
    return saved


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Build SkyClearAI processed tiles.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--clear-dir", type=Path, default=Path("data/raw/sentinel2_clear"))
    parser.add_argument(
        "--cloud-probability-dir",
        type=Path,
        default=Path("data/raw/sentinel2_cloud_probability"),
    )
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Ignore local GeoTIFF scenes and build the deterministic synthetic fixture.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the data pipeline entrypoint."""

    configure_logging()
    args = parse_args()
    cloud_probability_dir = args.cloud_probability_dir if args.cloud_probability_dir.exists() else None
    if args.force_synthetic or not args.clear_dir.exists():
        LOGGER.info("Using synthetic Sentinel-2 path.")
        build_synthetic_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            patch_size=args.patch_size,
            seed=args.seed,
        )
    else:
        LOGGER.info("Using local GeoTIFF scenes from %s", args.clear_dir)
        build_from_geotiffs(
            clear_dir=args.clear_dir,
            output_dir=args.output_dir,
            cloud_probability_dir=cloud_probability_dir,
            patch_size=args.patch_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

