# SkyClearAI - Operational SAR-Optical Fusion and Cloud Removal

SkyClearAI is a production-grade framework designed to reconstruct cloud-covered optical satellite imagery (Sentinel-2) by leveraging Synthetic Aperture Radar (SAR) data (Sentinel-1 GRD). It uses a deep learning U-Net generator with PatchGAN adversarial training (Model 1) and compares results against zero-shot inpainting baselines (Model 2). 

This project is tailored for real-world operations, featuring real data acquisition from STAC APIs, automated geographic co-registration, s2cloudless cloud mask extraction, and overlap-blended tiled reconstruction of large satellite scenes.

---

## Key Capabilities

1. **Real Data Acquisition**: Connects directly to the Element84 Earth Search STAC API to dynamically query, crop, and download paired clear/cloudy Sentinel-2 scenes and co-temporal Sentinel-1 GRD imagery using windowed COG reads.
2. **SAR-Optical Co-registration**: Warps Sentinel-1 SAR VV/VH polarization channels onto the exact coordinate system, bounding box, spatial transform, and dimensions of the optical grid using high-fidelity geographic reprojection.
3. **Real Cloud Masking**: Employs the s2cloudless pixel detector to extract realistic cloud shapes from genuinely cloudy scenes instead of resorting to simple procedural noise (e.g. Perlin noise).
4. **GeoTIFF Stitching and Overlap Blending**: Processes large satellite scenes by splitting them into overlapping tiles, running predictions, and re-assembling them back to a fully georeferenced output GeoTIFF using 2D feathering weights to eliminate seam artifacts.
5. **Interactive Web Application**: Streamlit dashboard designed to upload large cloudy optical and SAR GeoTIFFs, run tiled reconstructions, visualize predictions, and download stitched products.

---

## Project Structure

```text
skyclear-ai/
├── app/
│   └── streamlit_app.py        # Streamlit web interface for tiled inference and downloads
├── data/
│   ├── raw/                    # Downloaded Sentinel-1 and Sentinel-2 GeoTIFF files
│   └── processed/              # Tiled train/val/test NumPy .npz packages
├── docs/
│   ├── ARCHITECTURE.md         # Detailed architectural designs
│   └── README.md               # Quick setup guidelines
├── src/
│   ├── models/
│   │   ├── baseline_inpaint.py # Model 2 baseline adapters (LaMa, SD, Simple)
│   │   ├── discriminator.py    # PatchGAN discriminator for optical image realism
│   │   └── generator.py        # U-Net generator for SAR-fused reconstruction
│   ├── acquire_data.py         # STAC client for Sentinel-1 and 2 crop downloads
│   ├── alignment.py            # Co-registration utilities using rasterio.warp
│   ├── constants.py            # Canonical band-order and stacked channel constants
│   ├── data_pipeline.py        # Pre-processing, tiling, and data packing entrypoint
│   ├── evaluate.py             # Performance evaluator (masked PSNR, SSIM, SAM, NDVI)
│   ├── infer.py                # Standalone inference and ablation testing runner
│   ├── stitcher.py             # TiledPredictorStitcher with overlap seam blending
│   ├── synthetic_clouds.py     # Cloud transplanting and s2cloudless masking utilities
│   └── train.py                # Model 1 GAN training entrypoint
├── tests/
│   └── test_smoke.py           # Comprehensive unit and integration test suite
├── pyproject.toml              # Build configuration and project dependencies
└── requirements.txt            # Pinned requirements file
```

---

## Installation and Setup

Ensure you have Python 3.10+ installed. For environments requiring GDAL (such as Colab, Kaggle, or local servers), install the dependencies using the pinned requirements:

```bash
# Clone the repository and navigate inside
cd skyclear-ai

# Install the package and dev dependencies in editable mode
pip install -e .[dev]

# Alternatively, install using the requirements file
pip install -r requirements.txt
```

---

## Command Runner Interface

A Windows batch file (`run_skyclear.bat`) is included to automate all workflow steps.

```text
Usage:
  run_skyclear.bat <mode>

Modes:
  full       Run synthetic data pipeline, training, inference, evaluation, and launch app.
  real       Run pipeline using real Sentinel-1/2 data (downloads automatically if missing).
  download   Query STAC and crop Sentinel-1 GRD and Sentinel-2 clear/cloudy scenes.
  smoke      Run a fast synthetic CPU wiring check (no trained checkpoint required).
  test       Run automated unit and integration tests.
  setup      Install required Python packages and dependencies in editable mode.
  app        Launch the interactive Streamlit dashboard directly.

Configuration overrides (set as env variables):
  SKYCLEAR_BBOX               Target WGS84 bounding box (default: 11.5 48.1 11.6 48.2)
  SKYCLEAR_DATE               Target date range (default: 2023-06-01/2023-08-31)
  SKYCLEAR_NUM_SAMPLES        Number of synthetic samples (default: 24)
  SKYCLEAR_PATCH_SIZE         Tiled patch dimension in pixels (default: 256)
  SKYCLEAR_EPOCHS             Number of training epochs (default: 10)
  SKYCLEAR_BATCH_SIZE         Training batch size (default: 2)
  SKYCLEAR_BASE_CHANNELS      Generator/Discriminator width (default: 32)
  SKYCLEAR_LAUNCH_APP=0       Disables launching Streamlit after pipeline runs.
```

Example usage:
```cmd
# Set up dependencies
run_skyclear.bat setup

# Run smoke tests
run_skyclear.bat test

# Download real imagery crops
run_skyclear.bat download

# Run the complete pipeline on real Sentinel data
run_skyclear.bat real
```

---

## Step-by-Step Operational Workflow

### Step 1: Real Data Acquisition
Query and download real Sentinel-2 and Sentinel-1 imagery crops directly over a specific WGS84 bounding box (e.g., near Munich, Germany) and date range:

```bash
python -m src.acquire_data --bbox 11.5 48.1 11.6 48.2 --date 2023-06-01/2023-08-31 --output-dir data/raw
```
This saves:
- `data/raw/sentinel2_clear.tif` (4 optical bands: B02, B03, B04, B08)
- `data/raw/sentinel2_cloudy.tif` (10 bands for s2cloudless masking)
- `data/raw/sentinel1_grd.tif` (2 SAR polarizations: VV, VH)

### Step 2: Data Tiling and Preparation
Prepare training, validation, and test datasets. The pipeline tiles the raw scenes into 256x256 patches, extracts real cloud masks using s2cloudless, transplants them onto the clear scenes, and aligns the SAR channels:

```bash
python -m src.data_pipeline --clear-dir data/raw --patch-size 256 --output-dir data/processed
```
*Note: If no raw GeoTIFF scenes exist, the pipeline falls back to generating a synthetic dataset fixture using `--force-synthetic`.*

### Step 3: Model 1 Training
Train the SAR-fused U-Net generator against the PatchGAN discriminator using mask-weighted L1, adversarial, spectral angle mapper (SAM), and VGG feature losses:

```bash
python -m src.train --processed-dir data/processed --epochs 10 --batch-size 2 --checkpoint-dir checkpoints
```

### Step 4: Standalone Inference
Generate predictions on the test dataset split, outputting full SAR-fused reconstructions, SAR-ablated predictions (zeroed SAR input), and baseline inpaintings:

```bash
python -m src.infer --processed-dir data/processed --split test --checkpoint checkpoints/model1_latest.pt --output-dir outputs/inference
```

### Step 5: Metric Evaluation
Evaluate the reconstructions inside the cloud-mask region against the clear target imagery. Generates PSNR, SSIM, SAM, and NDVI delta reports:

```bash
python -m src.evaluate --inference-dir outputs/inference --output-dir outputs/evaluation
```

### Step 6: Interactive Dashboard
Launch the Streamlit web application to perform tiled inference and seam-blending on large custom GeoTIFF files:

```bash
streamlit run app/streamlit_app.py
```

---

## Data Contract and Constants

To prevent silent bugs across the codebase, band and channel ordering are managed strictly in `src/constants.py`:

*   **Sentinel-2 Optical Order**:
    *   `BAND_BLUE` = 0 (B02)
    *   `BAND_GREEN` = 1 (B03)
    *   `BAND_RED` = 2 (B04)
    *   `BAND_NIR` = 3 (B08)
*   **Sentinel-1 SAR Order**:
    *   `BAND_VV` = 0
    *   `BAND_VH` = 1
*   **Model 1 Generator Input (7 Channels)**:
    *   Channels 0-3: Cloudy Optical (Blue, Green, Red, NIR)
    *   Channels 4-5: Co-registered SAR (VV, VH)
    *   Channel 6: Cloud Mask

---

## Stitching and Overlap Blending

The `TiledPredictorStitcher` (in `src/stitcher.py`) avoids boundary edge artifacts by tiling large scenes with a user-defined pixel overlap (e.g. 64 pixels). During re-assembly, a 2D feathering weight matrix is applied:

$$W(x, y) = W_{1D}(x) \times W_{1D}(y)$$

Where $W_{1D}$ contains linear ramps from 0 to 1 over the overlap margins. The final value at pixel (x, y) is:

$$P_{final}(x, y) = \frac{\sum_{t} W_t(x, y) \cdot P_t(x, y)}{\sum_{t} W_t(x, y)}$$

This ensures transitions between neighboring patches are visually seamless, preserving the target CRS and transform exactly.

---

## Testing

To execute the unit and integration tests (including the Generator/Discriminator shape check, reprojection co-registration, and patch-stitching test):

```bash
pytest tests/test_smoke.py
```
