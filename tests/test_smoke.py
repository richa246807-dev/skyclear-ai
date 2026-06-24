"""Smoke tests for the scoped SkyClearAI build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_pipeline import build_synthetic_dataset
from src.evaluate import compute_ndvi, masked_psnr, save_ndvi_visualization
from src.infer import InferenceConfig, build_model_input, run_inference
from src.synthetic_clouds import generate_realistic_cloud_mask, mask_aware_composite


def test_cloud_mask_generation_is_deterministic_and_feathered() -> None:
    """Generated masks should be deterministic, bounded, and soft-edged."""

    mask_a = generate_realistic_cloud_mask(32, 32, seed=7, feather_radius=3)
    mask_b = generate_realistic_cloud_mask(32, 32, seed=7, feather_radius=3)
    assert mask_a.shape == (32, 32)
    assert np.allclose(mask_a, mask_b)
    assert float(mask_a.min()) >= 0.0
    assert float(mask_a.max()) <= 1.0
    assert np.any((mask_a > 0.0) & (mask_a < 1.0))


def test_data_pipeline_sample_contract_and_dataset(tmp_path: Path) -> None:
    """Processed samples should expose the expected arrays and 7-channel model input."""

    processed_dir = tmp_path / "processed"
    saved = build_synthetic_dataset(processed_dir, num_samples=6, patch_size=32, seed=10)
    assert len(saved) == 6
    with np.load(saved[0], allow_pickle=False) as sample:
        assert sample["cloudy_optical"].shape == (4, 32, 32)
        assert sample["target_optical"].shape == (4, 32, 32)
        assert sample["sar"].shape == (2, 32, 32)
        assert sample["cloud_mask"].shape == (32, 32)
        metadata = json.loads(str(sample["metadata_json"].tolist()))
    assert metadata["patch_size"] == 32

    pytest.importorskip("torch")
    from src.train import ProcessedTileDataset

    dataset = ProcessedTileDataset(processed_dir, split="train")
    item = dataset[0]
    assert item["input"].shape[0] == 7
    assert item["target"].shape[0] == 4


def test_mask_aware_composite_preserves_non_cloud_pixels() -> None:
    """Compositing should leave pixels outside the cloud mask unchanged."""

    original = np.zeros((4, 8, 8), dtype=np.float32)
    generated = np.ones((4, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    composite = mask_aware_composite(original, generated, mask, feather_radius=0)
    assert np.allclose(composite[:, mask == 0.0], original[:, mask == 0.0])
    assert np.allclose(composite[:, mask == 1.0], generated[:, mask == 1.0])


def test_masked_metrics_ignore_unmasked_pixels() -> None:
    """Masked PSNR should not be affected by errors outside the cloud mask."""

    target = np.zeros((4, 8, 8), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:, 0, 0] = 1.0
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    assert masked_psnr(prediction, target, mask) == float("inf")


def test_ndvi_visualization_writes_file(tmp_path: Path) -> None:
    """NDVI visualization should be produced without ground truth."""

    pytest.importorskip("matplotlib")
    optical = np.zeros((4, 8, 8), dtype=np.float32)
    optical[2] = 0.2
    optical[3] = 0.6
    ndvi = compute_ndvi(optical)
    assert np.allclose(ndvi, 0.5, atol=1e-6)
    output_path = tmp_path / "ndvi.png"
    save_ndvi_visualization(output_path, optical, optical)
    assert output_path.exists()


def test_sar_zero_input_path() -> None:
    """SAR ablation should zero only the SAR channels."""

    cloudy = np.ones((4, 8, 8), dtype=np.float32)
    sar = np.ones((2, 8, 8), dtype=np.float32) * 0.5
    mask = np.ones((8, 8), dtype=np.float32)
    normal = build_model_input(cloudy, sar, mask, zero_sar=False)
    ablated = build_model_input(cloudy, sar, mask, zero_sar=True)
    assert np.allclose(normal[4:6], 0.5)
    assert np.allclose(ablated[4:6], 0.0)
    assert np.allclose(ablated[:4], cloudy)
    assert np.allclose(ablated[6], mask)


def test_inference_entrypoint_smoke(tmp_path: Path) -> None:
    """Inference should run end-to-end on a tiny fixture with smoke settings."""

    pytest.importorskip("torch")
    processed_dir = tmp_path / "processed"
    build_synthetic_dataset(processed_dir, num_samples=6, patch_size=32, seed=12)
    output_dir = tmp_path / "inference"
    saved = run_inference(
        InferenceConfig(
            processed_dir=processed_dir,
            split="test",
            checkpoint=tmp_path / "missing.pt",
            output_dir=output_dir,
            baseline_backend="simple",
            lama_command=None,
            stable_diffusion_model=None,
            allow_random_weights=True,
            allow_simple_baseline=True,
            base_channels=4,
            device="cpu",
            limit=1,
        )
    )
    assert len(saved) == 1
    with np.load(saved[0], allow_pickle=False) as sample:
        assert sample["model1_output"].shape == (4, 32, 32)
        assert sample["model1_sar_zero_output"].shape == (4, 32, 32)
        assert sample["model2_output"].shape == (4, 32, 32)


def test_model_shapes() -> None:
    """Shape smoke test for the Generator and Discriminator forward pass."""

    pytest.importorskip("torch")
    import torch
    from src.models.generator import UNetGenerator
    from src.models.discriminator import PatchGANDiscriminator

    gen = UNetGenerator(in_channels=7, out_channels=4, base_channels=8)
    dummy_in = torch.randn(1, 7, 256, 256)
    dummy_out = gen(dummy_in)
    assert dummy_out.shape == (1, 4, 256, 256)

    disc = PatchGANDiscriminator(in_channels=4, base_channels=8)
    dummy_disc_in = torch.randn(1, 4, 256, 256)
    dummy_disc_out = disc(dummy_disc_in)
    assert dummy_disc_out.ndim == 4
    assert dummy_disc_out.shape[1] == 1


def test_reprojection_alignment(tmp_path: Path) -> None:
    """Test raster alignment via reprojection."""

    import rasterio
    from rasterio.transform import from_origin
    from src.alignment import reproject_to_reference

    ref_path = tmp_path / "reference.tif"
    ref_transform = from_origin(11.5, 48.2, 0.01, 0.01)
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=ref_transform,
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype=np.float32) * 5.0)

    src_path = tmp_path / "source.tif"
    src_transform = from_origin(11.4, 48.3, 0.01, 0.01)
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=30,
        width=30,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=src_transform,
    ) as dst:
        dst.write(np.ones((2, 30, 30), dtype=np.float32) * 10.0)

    aligned = reproject_to_reference(src_path, ref_path)
    assert aligned.shape == (2, 10, 10)
    assert np.allclose(aligned, 10.0, atol=1e-5)


def test_tiled_predictor_stitcher(tmp_path: Path) -> None:
    """Test that the tiled predictor stitcher runs and preserves CRS/transforms."""

    pytest.importorskip("torch")
    import rasterio
    from rasterio.transform import from_origin
    from src.models.generator import build_generator
    from src.models.baseline_inpaint import BaselineConfig, BaselineInpainter
    from src.stitcher import TiledPredictorStitcher

    opt_path = tmp_path / "optical.tif"
    opt_transform = from_origin(11.5, 48.2, 0.01, 0.01)
    with rasterio.open(
        opt_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=opt_transform,
    ) as dst:
        dst.write(np.random.rand(4, 64, 64).astype(np.float32))

    sar_path = tmp_path / "sar.tif"
    with rasterio.open(
        sar_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=opt_transform,
    ) as dst:
        dst.write(np.random.rand(2, 64, 64).astype(np.float32))

    model1 = build_generator(base_channels=4)
    baseline = BaselineInpainter(BaselineConfig(backend="simple"), device="cpu")
    stitcher = TiledPredictorStitcher(
        model1=model1,
        baseline=baseline,
        device="cpu",
        patch_size=32,
        overlap=8,
    )

    outputs = stitcher.process_and_stitch(opt_path, sar_path, None, tmp_path / "output")
    assert "model1_output" in outputs
    assert "model2_output" in outputs
    assert "cloud_mask" in outputs

    with rasterio.open(outputs["model1_output"]) as src:
        assert src.crs == rasterio.crs.CRS.from_epsg(4326)
        assert src.transform == opt_transform
        assert src.shape == (64, 64)
        assert src.count == 4


