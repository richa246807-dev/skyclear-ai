"""Single entrypoint for metrics, NDVI visualization, and SAR-ablation reporting."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


from src.constants import BAND_NIR, BAND_RED

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricRow:
    """Per-sample metric row.

    Args:
        sample_id: Sample identifier.
        model: Model label.
        psnr: Masked PSNR.
        ssim: Masked SSIM-like statistic.
        sam_degrees: Masked spectral angle mapper in degrees.
    """

    sample_id: str
    model: str
    psnr: float
    ssim: float
    sam_degrees: float


@dataclass(frozen=True)
class SarDeltaRow:
    """Per-sample SAR-ablation delta row."""

    sample_id: str
    psnr_delta: float
    ssim_delta: float
    sam_delta_degrees: float


def _mask_bool(mask: np.ndarray) -> np.ndarray:
    """Return a hard boolean mask from a soft cloud mask."""

    if mask.ndim != 2:
        raise ValueError("Cloud mask must have shape (height, width).")
    hard = mask > 0.2
    if not np.any(hard):
        raise ValueError("Cloud mask contains no positive pixels.")
    return hard


def masked_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    cloud_mask: np.ndarray,
    max_value: float = 1.0,
) -> float:
    """Compute PSNR over cloud-mask pixels only."""

    hard_mask = _mask_bool(cloud_mask)
    diff = prediction[:, hard_mask] - target[:, hard_mask]
    mse = float(np.mean(diff**2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def masked_ssim(prediction: np.ndarray, target: np.ndarray, cloud_mask: np.ndarray) -> float:
    """Compute an SSIM-style statistic over cloud-mask pixels only."""

    hard_mask = _mask_bool(cloud_mask)
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for band in range(prediction.shape[0]):
        x = prediction[band][hard_mask].astype(np.float64)
        y = target[band][hard_mask].astype(np.float64)
        mux = float(np.mean(x))
        muy = float(np.mean(y))
        varx = float(np.var(x))
        vary = float(np.var(y))
        covxy = float(np.mean((x - mux) * (y - muy)))
        numerator = (2.0 * mux * muy + c1) * (2.0 * covxy + c2)
        denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
        scores.append(numerator / denominator)
    return float(np.mean(scores))


def masked_sam_degrees(
    prediction: np.ndarray,
    target: np.ndarray,
    cloud_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute spectral angle mapper over cloud-mask pixels only."""

    hard_mask = _mask_bool(cloud_mask)
    pred_vectors = prediction[:, hard_mask].T.astype(np.float64)
    target_vectors = target[:, hard_mask].T.astype(np.float64)
    dot = np.sum(pred_vectors * target_vectors, axis=1)
    pred_norm = np.linalg.norm(pred_vectors, axis=1)
    target_norm = np.linalg.norm(target_vectors, axis=1)
    cosine = dot / np.maximum(pred_norm * target_norm, eps)
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.mean(angles)))


def compute_ndvi(optical: np.ndarray, red_index: int = BAND_RED, nir_index: int = BAND_NIR) -> np.ndarray:
    """Compute NDVI from channel-first optical data."""

    if optical.ndim != 3:
        raise ValueError("Optical tile must have shape (bands, height, width).")
    if optical.shape[0] <= max(red_index, nir_index):
        raise ValueError("Optical tile does not contain the requested red and NIR bands.")
    red = optical[red_index]
    nir = optical[nir_index]
    return ((nir - red) / np.maximum(nir + red, 1e-6)).astype(np.float32)


def save_ndvi_visualization(
    output_path: Path,
    before_optical: np.ndarray,
    after_optical: np.ndarray,
) -> None:
    """Save a side-by-side before and after NDVI visualization."""

    import matplotlib.pyplot as plt

    before = compute_ndvi(before_optical)
    after = compute_ndvi(after_optical)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, data, title in zip(axes, [before, after], ["Before", "After"], strict=True):
        image = axis.imshow(data, vmin=-1.0, vmax=1.0, cmap="RdYlGn")
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="NDVI")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_metadata(raw_metadata: np.ndarray | str) -> dict[str, object]:
    """Load metadata JSON from an ``.npz`` value."""

    if isinstance(raw_metadata, np.ndarray):
        raw = str(raw_metadata.tolist())
    else:
        raw = raw_metadata
    return json.loads(raw)


def evaluate_prediction(
    sample_id: str,
    model: str,
    prediction: np.ndarray,
    target: np.ndarray,
    cloud_mask: np.ndarray,
) -> MetricRow:
    """Evaluate one model output against a target."""

    return MetricRow(
        sample_id=sample_id,
        model=model,
        psnr=masked_psnr(prediction, target, cloud_mask),
        ssim=masked_ssim(prediction, target, cloud_mask),
        sam_degrees=masked_sam_degrees(prediction, target, cloud_mask),
    )


def _finite_mean(values: list[float]) -> float | None:
    """Return finite mean or ``None`` if no finite values are available."""

    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def _json_safe(value: float | None) -> float | None:
    """Return a JSON-safe float value."""

    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return value


def write_metric_csv(path: Path, rows: list[MetricRow]) -> None:
    """Write metric rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_delta_csv(path: Path, rows: list[SarDeltaRow]) -> None:
    """Write SAR-ablation delta rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def evaluate_inference_directory(inference_dir: Path, output_dir: Path) -> dict[str, object]:
    """Evaluate all inference archives and write reports."""

    paths = sorted(inference_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No inference outputs found in {inference_dir}.")

    metric_rows: list[MetricRow] = []
    delta_rows: list[SarDeltaRow] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            metadata = load_metadata(sample["metadata_json"])
            sample_id = str(metadata.get("sample_id", path.stem))
            cloudy = sample["cloudy_optical"].astype(np.float32)
            target = sample["target_optical"].astype(np.float32)
            mask = sample["cloud_mask"].astype(np.float32)
            model1 = sample["model1_output"].astype(np.float32)
            model1_zero = sample["model1_sar_zero_output"].astype(np.float32)
            model2 = sample["model2_output"].astype(np.float32)

        model1_row = evaluate_prediction(sample_id, "model1_sar", model1, target, mask)
        zero_row = evaluate_prediction(sample_id, "model1_sar_zero", model1_zero, target, mask)
        model2_row = evaluate_prediction(sample_id, "model2_baseline", model2, target, mask)
        metric_rows.extend([model1_row, zero_row, model2_row])
        delta_rows.append(
            SarDeltaRow(
                sample_id=sample_id,
                psnr_delta=model1_row.psnr - zero_row.psnr,
                ssim_delta=model1_row.ssim - zero_row.ssim,
                sam_delta_degrees=model1_row.sam_degrees - zero_row.sam_degrees,
            )
        )
        save_ndvi_visualization(output_dir / "ndvi" / f"{sample_id}_model1.png", cloudy, model1)

    write_metric_csv(output_dir / "metrics_table.csv", metric_rows)
    write_delta_csv(output_dir / "sar_ablation_delta.csv", delta_rows)

    summary: dict[str, Any] = {"models": {}, "sar_ablation_delta": {}}
    for model_name in sorted({row.model for row in metric_rows}):
        rows = [row for row in metric_rows if row.model == model_name]
        summary["models"][model_name] = {
            "psnr": _json_safe(_finite_mean([row.psnr for row in rows])),
            "ssim": _json_safe(_finite_mean([row.ssim for row in rows])),
            "sam_degrees": _json_safe(_finite_mean([row.sam_degrees for row in rows])),
        }
    summary["sar_ablation_delta"] = {
        "psnr_delta": _json_safe(_finite_mean([row.psnr_delta for row in delta_rows])),
        "ssim_delta": _json_safe(_finite_mean([row.ssim_delta for row in delta_rows])),
        "sam_delta_degrees": _json_safe(_finite_mean([row.sam_delta_degrees for row in delta_rows])),
    }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    LOGGER.info("Wrote evaluation outputs under %s", output_dir)
    return summary


def configure_logging() -> None:
    """Configure process logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate SkyClearAI inference outputs.")
    parser.add_argument("--inference-dir", type=Path, default=Path("outputs/inference/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    return parser.parse_args()


def main() -> None:
    """Run the evaluation entrypoint."""

    configure_logging()
    args = parse_args()
    evaluate_inference_directory(args.inference_dir, args.output_dir)


if __name__ == "__main__":
    main()

