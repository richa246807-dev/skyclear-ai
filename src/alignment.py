"""Utilities for spatial co-registration and aligning SAR and optical rasters."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

LOGGER = logging.getLogger(__name__)


def reproject_to_reference(
    src_path: Path | str,
    ref_path: Path | str,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject a source raster to match the grid of a reference raster.

    Args:
        src_path: Path to the source raster (to be warped).
        ref_path: Path to the reference raster defining the target grid.
        resampling: The resampling algorithm to use.

    Returns:
        Warped raster data as a numpy array with shape (bands, ref_height, ref_width).
    """
    LOGGER.info("Aligning source raster %s to reference grid %s", src_path, ref_path)
    
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_dtype = ref.dtypes[0]

    with rasterio.open(src_path) as src:
        src_count = src.count
        src_dtype = src.dtypes[0]
        # Allocate destination array matching reference size and source band count
        dst_array = np.zeros((src_count, dst_height, dst_width), dtype=np.float32)

        for band_idx in range(1, src_count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_array[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )

    return dst_array


def align_and_save(
    src_path: Path | str,
    ref_path: Path | str,
    output_path: Path | str,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject a source raster to match a reference raster and save it as a new GeoTIFF."""
    dst_array = reproject_to_reference(src_path, ref_path, resampling=resampling)
    
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        
    with rasterio.open(src_path) as src:
        src_count = src.count
        src_dtype = src.dtypes[0]

    profile.update({
        "count": src_count,
        "dtype": src_dtype,
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving aligned raster to %s", output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dst_array)
