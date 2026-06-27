"""Streamlit demo entrypoint for SkyClearAI GeoTIFF tiled inference and stitching."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_prediction  # noqa: E402
from src.infer import load_model1  # noqa: E402
from src.models.baseline_inpaint import BaselineConfig, BaselineInpainter  # noqa: E402
from src.stitcher import TiledPredictorStitcher  # noqa: E402


def save_uploaded_to_temp(uploaded_file: Any) -> Path:
    """Save a Streamlit uploaded file to a temporary location and return the Path."""
    suffix = Path(uploaded_file.name).suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        return Path(handle.name)

def optical_to_rgb(optical: np.ndarray) -> np.ndarray:
    from src.constants import BAND_RED, BAND_GREEN, BAND_BLUE

    if optical.shape[0] >= 4:
        rgb = np.stack([
            optical[BAND_RED],
            optical[BAND_GREEN],
            optical[BAND_BLUE]
        ], axis=-1)
    else:
        rgb = np.moveaxis(optical[:3], 0, -1)

    rgb = rgb.astype(np.float32)

    # Contrast Stretch
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)

    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    # Gamma Correction
    gamma = 0.9
    rgb = rgb ** gamma

    return rgb


def read_and_preview_raster(path: Path, max_dim: int = 512) -> np.ndarray:
    """Read a raster with optional downsampling for preview visualization."""
    with rasterio.open(path) as src:
        ratio = max(1.0, max(src.height, src.width) / max_dim)
        if ratio > 1.0:
            out_shape = (src.count, int(src.height / ratio), int(src.width / ratio))
            data = src.read(
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.bilinear,
            )
        else:
            data = src.read()
    return data.astype(np.float32)


def main() -> None:
    """Run the Streamlit app."""
    import torch
    import streamlit as st

    st.set_page_config(
    page_title="SkyClearAI",
    page_icon="🛰️",
    layout="wide"
)

    st.title("🛰️ SkyClearAI")

    st.markdown("""
### AI-Powered Satellite Cloud Removal & Earth Observation

Reconstruct cloud-covered Sentinel-2 imagery using **SAR + Optical Fusion Deep Learning**.

**ISRO Bharatiya Antariksh Hackathon 2026 Prototype**
""")

    st.divider()

    # Sidebar inputs
    st.sidebar.header("Configuration")
    checkpoint_path = Path(st.sidebar.text_input("Model 1 Checkpoint", "checkpoints/model1_latest.pt"))
    patch_size = st.sidebar.number_input("Patch Size", min_value=64, max_value=512, value=256, step=64)
    overlap = st.sidebar.number_input("Patch Overlap", min_value=0, max_value=256, value=64, step=16)
    base_channels = st.sidebar.number_input("Base Channels", min_value=4, max_value=64, value=32, step=4)

    baseline_backend = st.sidebar.selectbox("Baseline Backend", ["auto", "lama", "stable-diffusion", "simple"])
    allow_random = st.sidebar.checkbox("Allow untrained weights (for testing)", value=True)

    # File uploads
    st.subheader("Data Uploads")
    col_upload_1, col_upload_2, col_upload_3 = st.columns(3)
    
    uploaded_opt = col_upload_1.file_uploader("Cloudy Optical GeoTIFF (e.g., sentinel2_cloudy.tif)", type=["tif", "tiff"])
    uploaded_sar = col_upload_2.file_uploader("SAR GeoTIFF (e.g., sentinel1_grd.tif)", type=["tif", "tiff"])
    uploaded_mask = col_upload_3.file_uploader("Cloud Mask GeoTIFF (Optional, e.g., cloud_mask.tif)", type=["tif", "tiff"])

    if uploaded_opt is None or uploaded_sar is None:
        st.info("Please upload both 'sentinel2_cloudy.tif' (Cloudy Optical) and 'sentinel1_grd.tif' (SAR) from your 'data/raw/' directory to begin.")
        st.stop()

    if st.button("Run Reconstruction & Stitching", type="primary"):
        # Save uploaded files to temp
        opt_temp_path = save_uploaded_to_temp(uploaded_opt)
        sar_temp_path = save_uploaded_to_temp(uploaded_sar)
        mask_temp_path = save_uploaded_to_temp(uploaded_mask) if uploaded_mask else None

        # Validate band counts before starting
        try:
            with rasterio.open(opt_temp_path) as src:
                opt_count = src.count
            with rasterio.open(sar_temp_path) as src:
                sar_count = src.count
        except Exception as e:
            st.error(f"Error opening uploaded GeoTIFF files: {e}")
            st.stop()

        if opt_count < 4:
            st.error(f"The uploaded Cloudy Optical GeoTIFF has only {opt_count} band(s). It must have at least 4 bands (Blue, Green, Red, NIR). Please make sure you did not upload the SAR image (sentinel1_grd.tif) in the Optical slot.")
            opt_temp_path.unlink(missing_ok=True)
            sar_temp_path.unlink(missing_ok=True)
            if mask_temp_path:
                mask_temp_path.unlink(missing_ok=True)
            st.stop()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner("Loading models and preparing stitcher..."):
            try:
                model1 = load_model1(
                    checkpoint_path,
                    device=device,
                    base_channels=base_channels,
                    allow_random_weights=allow_random,
                )
                baseline = BaselineInpainter(
                    BaselineConfig(backend=baseline_backend, allow_simple_fallback=True),
                    device=device,
                )
                stitcher = TiledPredictorStitcher(
                    model1=model1,
                    baseline=baseline,
                    device=device,
                    patch_size=patch_size,
                    overlap=overlap,
                )
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

        with st.spinner("Performing tiled reconstruction and blending overlap seams..."):
            try:
                temp_output_dir = Path(tempfile.mkdtemp(prefix="skyclear_outputs_"))
                stitched_outputs = stitcher.process_and_stitch(
                    optical_path=opt_temp_path,
                    sar_path=sar_temp_path,
                    cloud_mask_path=mask_temp_path,
                    output_dir=temp_output_dir,
                )
                st.success("Reconstruction complete!")
                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Status", "Completed ✅")
                col2.metric("Patch Size", patch_size)
                col3.metric("Device", device.upper())
                col4.metric("Model", "SkyClearAI")
            except Exception as e:
                st.error(f"Error during reconstruction: {e}")
                st.stop()

        # Load previews
        opt_preview = read_and_preview_raster(opt_temp_path)
    
        mask_preview = read_and_preview_raster(stitched_outputs["cloud_mask"])[0]
        m1_preview = read_and_preview_raster(stitched_outputs["model1_output"])
        m2_preview = read_and_preview_raster(stitched_outputs["model2_output"])
        
        # Display results
        st.divider()
        st.header("🖼 Reconstruction Results")
        st.subheader("Visual Preview")
        col_res_1, col_res_2, col_res_3, col_res_4 = st.columns(4)
        col_res_1.image(optical_to_rgb(opt_preview), caption="☁️ Cloudy Optical Image", use_container_width=True)
        col_res_2.image(mask_preview, caption="☁️ Estimated Cloud Mask", clamp=True, use_container_width=True)
        col_res_3.image(optical_to_rgb(m1_preview), caption="🤖 AI Reconstruction (SAR Fusion) Stitched", use_container_width=True)
        col_res_4.image(optical_to_rgb(m2_preview), caption="🧩 Baseline Reconstruction Stitched", use_container_width=True)

        # Download Buttons
        st.divider()
        st.header("📥 Download Products")
        st.subheader("Download Reconstructed Products")
        col_dl_1, col_dl_2, col_dl_3 = st.columns(3)
        
        # M1 Download
        with open(stitched_outputs["model1_output"], "rb") as f:
            col_dl_1.download_button(
                label="📥 Download Model 1 (SAR Fusion) GeoTIFF",
                data=f.read(),
                file_name=f"{Path(uploaded_opt.name).stem}_reconstructed_model1.tif",
                mime="image/tiff",
            )
            
        # M2 Download
        with open(stitched_outputs["model2_output"], "rb") as f:
            col_dl_2.download_button(
                label="📥 Download Model 2 (Baseline) GeoTIFF",
                data=f.read(),
                file_name=f"{Path(uploaded_opt.name).stem}_reconstructed_model2.tif",
                mime="image/tiff",
            )

        # Mask Download
        with open(stitched_outputs["cloud_mask"], "rb") as f:
            col_dl_3.download_button(
                label="📥 Download Cloud Mask GeoTIFF",
                data=f.read(),
                file_name=f"{Path(uploaded_opt.name).stem}_cloud_mask.tif",
                mime="image/tiff",
            )

        # Clean up temp inputs
        opt_temp_path.unlink(missing_ok=True)
        sar_temp_path.unlink(missing_ok=True)
        if mask_temp_path:
            mask_temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
