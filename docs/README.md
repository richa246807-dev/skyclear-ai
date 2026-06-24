# SkyClearAI

SkyClearAI reconstructs cloud-covered optical satellite tiles with a SAR-fused U-Net and compares the result against a zero-shot inpainting baseline. The implementation follows the sprint scope only: data pipeline, Model 1 training, Model 2 inference, mask-aware compositing, evaluation, SAR ablation, and a Streamlit GeoTIFF demo.

## Setup

Use an environment with PyTorch, Rasterio/GDAL, and Streamlit available. On Colab or Kaggle, install the Python dependencies in `pyproject.toml` and keep checkpoints in mounted storage.

```bash
pip install -e ".[dev]"
```

## Stage Entrypoints

Run every stage from the `skyclear-ai/` directory.

```bash
python -m src.data_pipeline --force-synthetic --num-samples 24 --patch-size 256
python -m src.train --epochs 10 --batch-size 2 --checkpoint-every-steps 100
python -m src.infer --split test --checkpoint checkpoints/model1_latest.pt
python -m src.evaluate --inference-dir outputs/inference/test --output-dir outputs/evaluation
streamlit run app/streamlit_app.py
```

For a quick CPU smoke run, use smaller synthetic tiles and allow untrained inference only to verify wiring:

```bash
python -m src.data_pipeline --force-synthetic --num-samples 6 --patch-size 32 --output-dir data/processed_smoke
python -m src.infer --processed-dir data/processed_smoke --split test --output-dir outputs/smoke_inference --allow-random-weights --base-channels 4 --limit 1 --baseline-backend simple
python -m src.evaluate --inference-dir outputs/smoke_inference --output-dir outputs/smoke_evaluation
```

## Data Contract

Processed samples are compressed NumPy archives saved under `data/processed/{train,val,test}` with:

- `cloudy_optical`: four optical bands, channel-first, normalized to `[0, 1]`.
- `target_optical`: clear four-band target.
- `sar`: two channels for VV/VH-style SAR input.
- `cloud_mask`: soft mask where reconstruction is evaluated and composited.
- `metadata_json`: sample id, split, source, cloud fraction, and patch size.

If local Sentinel-2-like GeoTIFF scenes exist in `data/raw/sentinel2_clear`, the data pipeline tiles them. Otherwise it immediately builds the deterministic synthetic path. Bhoonidhi/LISS-IV access is not a blocker.

## Baseline Configuration

Model 2 is zero-shot. The preferred path is an external LaMa wrapper command:

```bash
python -m src.infer --baseline-backend lama --lama-command "python path/to/lama_wrapper.py"
```

The command is called with appended arguments: input image path, mask path, output image path. A local Stable Diffusion inpainting model can be selected with `--baseline-backend stable-diffusion --stable-diffusion-model <path-or-id>`. The simple backend is for offline smoke tests when pretrained baseline assets are not present.

## Evaluation

`src.evaluate` reports PSNR, SSIM, and SAM only inside the cloud mask. It writes:

- `metrics_table.csv`
- `sar_ablation_delta.csv`
- `metrics_summary.json`
- `ndvi/*.png`

The SAR delta is `model1_sar - model1_sar_zero` for each metric. For SAM, lower absolute values are better, so a negative SAM delta means the SAR-enabled run reduced spectral angle.

