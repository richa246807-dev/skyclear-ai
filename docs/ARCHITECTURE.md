# Architecture

## Data Flow

The default pipeline builds paired cloudy and cloud-free tiles from Sentinel-2-like optical data. Real cloud shapes come from cloud probability rasters when available; otherwise the pipeline estimates or synthesizes irregular cloud masks, feathers the mask boundary, applies opacity variation, and adds shifted shadows. The output is a fixed sample contract consumed by training, inference, evaluation, and the app.

## Model 1

Model 1 is the trained SAR-fusion model. Its input has seven channels: four cloudy optical bands, SAR VV/VH, and one cloud mask. The generator is a U-Net that emits four reconstructed optical bands. The discriminator is a PatchGAN over optical tiles. Training uses mask-weighted L1, adversarial loss, VGG feature loss, and spectral angle mapper loss.

Checkpoints contain both model optimizers and the current step/epoch so interrupted free-tier GPU sessions can resume.

## Model 2

Model 2 is a zero-shot inpainting baseline using optical input and the same cloud mask. LaMa is preferred through an external command adapter. Stable Diffusion inpainting is available as a fallback when a local model is configured. The simple backend exists only to verify the pipeline without pretrained baseline assets.

## Compositing And Metrics

Both model outputs are composited with the original cloudy optical tile. Generated pixels are used inside the cloud mask, and original pixels are preserved outside it with a small feather at the mask boundary.

Evaluation compares outputs to the synthetic clear target only inside the cloud mask. Full-image PSNR, SSIM, and SAM are not reported because unchanged pixels outside the mask would inflate the result.

## Demo

The Streamlit app accepts a GeoTIFF, extracts a 256x256 tile through Rasterio, estimates or reads a cloud mask, runs both models, and displays the input tile, cloud mask, Model 1 output, Model 2 output, and evaluation metrics. Metrics are loaded from `outputs/evaluation/metrics_summary.json` when available.

