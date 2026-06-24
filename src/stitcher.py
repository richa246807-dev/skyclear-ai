"""Utility for tiled inference and georeferenced output stitching with overlap blending."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.alignment import reproject_to_reference
from src.constants import CH_SAR_VV, CH_SAR_VH, CH_CLOUD_MASK
from src.synthetic_clouds import estimate_cloud_mask_from_optical, mask_aware_composite
from src.infer import build_model_input, predict_model1
from src.models.baseline_inpaint import BaselineInpainter

LOGGER = logging.getLogger(__name__)


def get_feather_weight(size: int, overlap: int) -> np.ndarray:
    """Generate a 2D feathering weight matrix with linear ramps on overlap regions."""
    w_1d = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        w_1d[:overlap] = ramp
        w_1d[-overlap:] = ramp[::-1]
    return np.outer(w_1d, w_1d)


class TiledPredictorStitcher:
    """Handles windowed tiling of large rasters, model predictions, and stitching with blending."""

    def __init__(
        self,
        model1: Any,
        baseline: BaselineInpainter,
        device: str = "cpu",
        patch_size: int = 256,
        overlap: int = 64,
    ) -> None:
        """Initialize the stitcher."""
        self.model1 = model1
        self.baseline = baseline
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.step = patch_size - overlap
        if self.step <= 0:
            raise ValueError("Overlap must be strictly smaller than patch_size.")

    def process_and_stitch(
        self,
        optical_path: Path | str,
        sar_path: Path | str,
        cloud_mask_path: Path | str | None,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Perform tiled inference and stitch back into georeferenced GeoTIFFs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        optical_path = Path(optical_path)
        LOGGER.info("Opening optical reference raster: %s", optical_path)
        with rasterio.open(optical_path) as ref:
            profile = ref.profile.copy()
            height = ref.height
            width = ref.width
            # Read optical bands (C, H, W)
            if ref.count >= 10:
                # Sentinel-2 raw/11-band scene: extract B02, B03, B04, B08 (indices 2, 3, 4, 6 in 1-based indexing)
                LOGGER.info("Detected Sentinel-2 multi-band scene (%d bands). Extracting B02, B03, B04, B08.", ref.count)
                optical_data = ref.read([2, 3, 4, 6]).astype(np.float32)
            else:
                # Fallback: read the first 4 bands
                optical_data = ref.read(list(range(1, min(5, ref.count + 1)))).astype(np.float32)
            
            # Normalize optical data to [0, 1]
            optical_data = optical_data / max(1.0, float(np.percentile(optical_data, 99.5)))
            optical_data = np.clip(optical_data, 0.0, 1.0)


        # 1. Reproject and align SAR to match optical reference grid
        sar_data = reproject_to_reference(sar_path, optical_path)
        # Normalize SAR
        sar_data = sar_data / max(1.0, float(np.percentile(sar_data, 99.5)))
        sar_data = np.clip(sar_data, 0.0, 1.0)

        # 2. Get/estimate cloud mask
        if cloud_mask_path is not None and Path(cloud_mask_path).exists():
            LOGGER.info("Loading cloud mask from %s", cloud_mask_path)
            with rasterio.open(cloud_mask_path) as mask_file:
                mask_data = mask_file.read(1).astype(np.float32)
                if np.max(mask_data) > 1.1:
                    mask_data = mask_data / 255.0
                mask_data = np.clip(mask_data, 0.0, 1.0)
        else:
            LOGGER.info("No cloud mask provided. Estimating mask from optical bands...")
            mask_data = estimate_cloud_mask_from_optical(optical_data)

        # Allocate accumulator buffers
        accum_m1 = np.zeros((4, height, width), dtype=np.float32)
        accum_m2 = np.zeros((4, height, width), dtype=np.float32)
        accum_weights = np.zeros((height, width), dtype=np.float32)

        # 2D feathering weight matrix
        W = get_feather_weight(self.patch_size, self.overlap)

        # Slide over the image grid
        row_coords = list(range(0, height - self.patch_size + 1, self.step))
        if row_coords[-1] + self.patch_size < height:
            row_coords.append(height - self.patch_size)
        elif height < self.patch_size:
            row_coords = [0]

        col_coords = list(range(0, width - self.patch_size + 1, self.step))
        if col_coords[-1] + self.patch_size < width:
            col_coords.append(width - self.patch_size)
        elif width < self.patch_size:
            col_coords = [0]

        LOGGER.info("Running tiled inference on %dx%d grid...", len(row_coords), len(col_coords))

        for r in row_coords:
            for c in col_coords:
                # Handle boundaries by extracting a full patch
                r_start = min(r, height - self.patch_size) if height >= self.patch_size else 0
                c_start = min(c, width - self.patch_size) if width >= self.patch_size else 0
                
                # Slices
                opt_patch = optical_data[:, r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]
                sar_patch = sar_data[:, r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]
                mask_patch = mask_data[r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]

                # Ensure dimensions are correct (in case image is smaller than patch_size)
                if opt_patch.shape[1] != self.patch_size or opt_patch.shape[2] != self.patch_size:
                    # Pad
                    opt_pad = np.zeros((4, self.patch_size, self.patch_size), dtype=np.float32)
                    sar_pad = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
                    mask_pad = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                    
                    ph, pw = opt_patch.shape[1], opt_patch.shape[2]
                    opt_pad[:, :ph, :pw] = opt_patch
                    sar_pad[:, :ph, :pw] = sar_patch
                    mask_pad[:ph, :pw] = mask_patch
                    
                    # Run predictions
                    m1_out_raw = predict_model1(self.model1, opt_pad, sar_pad, mask_pad, device=self.device)
                    m2_out_raw = self.baseline.inpaint(opt_pad, mask_pad)
                    
                    # Composite
                    m1_comp_raw = mask_aware_composite(opt_pad, m1_out_raw, mask_pad)
                    m2_comp_raw = mask_aware_composite(opt_pad, m2_out_raw, mask_pad)
                    
                    # Extract back
                    m1_comp = m1_comp_raw[:, :ph, :pw]
                    m2_comp = m2_comp_raw[:, :ph, :pw]
                    w_patch = W[:ph, :pw]
                else:
                    # Run predictions directly
                    m1_out_raw = predict_model1(self.model1, opt_patch, sar_patch, mask_patch, device=self.device)
                    m2_out_raw = self.baseline.inpaint(opt_patch, mask_patch)
                    
                    # Composite
                    m1_comp = mask_aware_composite(opt_patch, m1_out_raw, mask_patch)
                    m2_comp = mask_aware_composite(opt_patch, m2_out_raw, mask_patch)
                    w_patch = W

                # Accumulate weighted patches
                accum_m1[:, r_start : r_start + w_patch.shape[0], c_start : c_start + w_patch.shape[1]] += m1_comp * w_patch[None, :, :]
                accum_m2[:, r_start : r_start + w_patch.shape[0], c_start : c_start + w_patch.shape[1]] += m2_comp * w_patch[None, :, :]
                accum_weights[r_start : r_start + w_patch.shape[0], c_start : c_start + w_patch.shape[1]] += w_patch

        # Normalize accumulators
        # Safe division
        weights_safe = np.maximum(accum_weights, 1e-8)
        output_m1 = accum_m1 / weights_safe[None, :, :]
        output_m2 = accum_m2 / weights_safe[None, :, :]

        output_m1 = np.clip(output_m1, 0.0, 1.0)
        output_m2 = np.clip(output_m2, 0.0, 1.0)

        # 3. Save stitched outputs to georeferenced GeoTIFFs
        path_m1 = output_dir / f"{optical_path.stem}_reconstructed_model1.tif"
        path_m2 = output_dir / f"{optical_path.stem}_reconstructed_model2.tif"
        path_mask = output_dir / f"{optical_path.stem}_cloud_mask.tif"

        # Update profile for output files
        profile.update({
            "driver": "GTiff",
            "count": 4,
            "dtype": "float32",
            "height": height,
            "width": width,
        })

        LOGGER.info("Writing output GeoTIFFs...")
        with rasterio.open(path_m1, "w", **profile) as dst:
            dst.write(output_m1)
            
        with rasterio.open(path_m2, "w", **profile) as dst:
            dst.write(output_m2)

        profile.update({"count": 1})
        with rasterio.open(path_mask, "w", **profile) as dst:
            dst.write(mask_data, 1)

        LOGGER.info("Stitching complete.")
        return {
            "model1_output": path_m1,
            "model2_output": path_m2,
            "cloud_mask": path_mask,
        }
