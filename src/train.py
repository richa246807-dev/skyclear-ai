"""Single entrypoint for training the SAR-fused SkyClearAI generator."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.discriminator import PatchGANDiscriminator, build_discriminator
from src.models.generator import UNetGenerator, build_generator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration.

    Args:
        processed_dir: Directory containing processed split folders.
        split: Training split name.
        checkpoint_dir: Directory where checkpoints are written.
        resume: Optional checkpoint path to resume from.
        epochs: Number of epochs.
        batch_size: Batch size.
        learning_rate: Adam learning rate.
        base_channels: Model width multiplier.
        checkpoint_every_steps: Checkpoint interval in optimizer steps.
        num_workers: DataLoader worker count.
        l1_weight: Weight for mask-weighted L1 loss.
        adversarial_weight: Weight for adversarial generator loss.
        perceptual_weight: Weight for VGG perceptual loss.
        sam_weight: Weight for spectral angle mapper loss.
        cloud_l1_multiplier: Extra L1 weight inside cloud mask.
        pretrained_vgg: Whether to request torchvision pretrained VGG weights.
        device: Torch device string.
    """

    processed_dir: Path
    split: str
    checkpoint_dir: Path
    resume: Path | None
    epochs: int
    batch_size: int
    learning_rate: float
    base_channels: int
    checkpoint_every_steps: int
    num_workers: int
    l1_weight: float
    adversarial_weight: float
    perceptual_weight: float
    sam_weight: float
    cloud_l1_multiplier: float
    pretrained_vgg: bool
    device: str


class ProcessedTileDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for processed SkyClearAI ``.npz`` tiles."""

    def __init__(self, processed_dir: Path, split: str = "train") -> None:
        """Initialize the dataset.

        Args:
            processed_dir: Root processed directory.
            split: Split name.

        Raises:
            FileNotFoundError: If the split directory has no samples.
        """

        self.split_dir = processed_dir / split
        self.paths = sorted(self.split_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No processed samples found in {self.split_dir}.")

    def __len__(self) -> int:
        """Return sample count."""

        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sample and return tensors for training."""

        path = self.paths[index]
        with np.load(path, allow_pickle=False) as sample:
            cloudy = sample["cloudy_optical"].astype(np.float32)
            target = sample["target_optical"].astype(np.float32)
            sar = sample["sar"].astype(np.float32)
            mask = sample["cloud_mask"].astype(np.float32)
        model_input = np.concatenate([cloudy, sar, mask[None, :, :]], axis=0)
        return {
            "input": torch.from_numpy(model_input),
            "target": torch.from_numpy(target),
            "cloudy_optical": torch.from_numpy(cloudy),
            "sar": torch.from_numpy(sar),
            "cloud_mask": torch.from_numpy(mask[None, :, :]),
        }


class VGGPerceptualLoss(nn.Module):
    """VGG feature-space L1 loss for optical RGB bands."""

    def __init__(self, pretrained: bool = False) -> None:
        """Initialize VGG features.

        Args:
            pretrained: Whether to request torchvision pretrained VGG16 weights.
        """

        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:
            raise RuntimeError("torchvision is required for VGG perceptual loss.") from exc

        weights = None
        if pretrained:
            try:
                weights = VGG16_Weights.DEFAULT
            except Exception:
                LOGGER.warning("Pretrained VGG weights are unavailable; using VGG architecture weights.")
                weights = None
        vgg = vgg16(weights=weights)
        self.features = vgg.features[:9].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _to_rgb(tensor: torch.Tensor) -> torch.Tensor:
        """Convert optical bands to RGB-like input for VGG."""

        from src.constants import BAND_RED, BAND_GREEN, BAND_BLUE
        if tensor.shape[1] >= 4:
            return tensor[:, [BAND_RED, BAND_GREEN, BAND_BLUE], :, :]
        if tensor.shape[1] == 3:
            return tensor
        return tensor.repeat(1, 3, 1, 1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute feature L1 distance."""

        prediction_rgb = self._to_rgb(prediction)
        target_rgb = self._to_rgb(target)
        prediction_features = self.features(prediction_rgb)
        target_features = self.features(target_rgb)
        return nn.functional.l1_loss(prediction_features, target_features)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: torch.Tensor,
    cloud_multiplier: float = 4.0,
) -> torch.Tensor:
    """Compute L1 loss with higher weight inside the cloud mask."""

    weight = 1.0 + cloud_multiplier * cloud_mask
    return torch.mean(torch.abs(prediction - target) * weight)


def spectral_angle_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean spectral angle mapper loss inside the cloud mask."""

    dot = torch.sum(prediction * target, dim=1)
    pred_norm = torch.linalg.norm(prediction, dim=1)
    target_norm = torch.linalg.norm(target, dim=1)
    cosine = dot / torch.clamp(pred_norm * target_norm, min=eps)
    angle = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
    mask = cloud_mask[:, 0]
    denominator = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(angle * mask) / denominator


def save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    """Save a resumable training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "config": json.dumps({key: str(value) for key, value in asdict(config).items()}),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[int, int]:
    """Load a checkpoint and return ``(step, epoch)``."""

    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))


def train_model(config: TrainConfig) -> Path:
    """Train Model 1 and return the final checkpoint path."""

    dataset = ProcessedTileDataset(config.processed_dir, config.split)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    device = torch.device(config.device)
    generator = build_generator(base_channels=config.base_channels).to(device)
    discriminator = build_discriminator(base_channels=config.base_channels).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(0.5, 0.999),
    )
    adversarial_loss = nn.BCEWithLogitsLoss()
    perceptual_loss = VGGPerceptualLoss(pretrained=config.pretrained_vgg).to(device)

    step = 0
    start_epoch = 0
    if config.resume is not None:
        step, start_epoch = load_checkpoint(
            config.resume,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            config.device,
        )
        LOGGER.info("Resumed checkpoint %s at step %d, epoch %d", config.resume, step, start_epoch)

    for epoch in range(start_epoch, config.epochs):
        generator.train()
        discriminator.train()
        for batch in loader:
            model_input = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["cloud_mask"].to(device)

            with torch.no_grad():
                fake_detached = generator(model_input).detach()
            real_logits = discriminator(target)
            fake_logits = discriminator(fake_detached)
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            d_loss = 0.5 * (
                adversarial_loss(real_logits, real_labels)
                + adversarial_loss(fake_logits, fake_labels)
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            discriminator_optimizer.step()

            prediction = generator(model_input)
            g_logits = discriminator(prediction)
            adv = adversarial_loss(g_logits, torch.ones_like(g_logits))
            l1 = masked_l1_loss(prediction, target, mask, config.cloud_l1_multiplier)
            sam = spectral_angle_loss(prediction, target, mask)
            perceptual = perceptual_loss(prediction, target)
            g_loss = (
                config.l1_weight * l1
                + config.adversarial_weight * adv
                + config.perceptual_weight * perceptual
                + config.sam_weight * sam
            )
            generator_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            generator_optimizer.step()

            step += 1
            if step % 10 == 0:
                LOGGER.info(
                    "epoch=%d step=%d d_loss=%.4f g_loss=%.4f l1=%.4f sam=%.4f",
                    epoch + 1,
                    step,
                    float(d_loss.detach().cpu()),
                    float(g_loss.detach().cpu()),
                    float(l1.detach().cpu()),
                    float(sam.detach().cpu()),
                )
            if step % config.checkpoint_every_steps == 0:
                checkpoint_path = config.checkpoint_dir / f"model1_step_{step:07d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    step,
                    epoch,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    config,
                )
                LOGGER.info("Saved checkpoint %s", checkpoint_path)

    final_path = config.checkpoint_dir / "model1_latest.pt"
    save_checkpoint(
        final_path,
        step,
        config.epochs,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        config,
    )
    LOGGER.info("Saved final checkpoint %s", final_path)
    return final_path


def configure_logging() -> None:
    """Configure process logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train SkyClearAI Model 1.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--l1-weight", type=float, default=10.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.05)
    parser.add_argument("--perceptual-weight", type=float, default=0.01)
    parser.add_argument("--sam-weight", type=float, default=0.2)
    parser.add_argument("--cloud-l1-multiplier", type=float, default=4.0)
    parser.add_argument("--pretrained-vgg", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training entrypoint."""

    configure_logging()
    args = parse_args()
    config = TrainConfig(
        processed_dir=args.processed_dir,
        split=args.split,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        checkpoint_every_steps=args.checkpoint_every_steps,
        num_workers=args.num_workers,
        l1_weight=args.l1_weight,
        adversarial_weight=args.adversarial_weight,
        perceptual_weight=args.perceptual_weight,
        sam_weight=args.sam_weight,
        cloud_l1_multiplier=args.cloud_l1_multiplier,
        pretrained_vgg=args.pretrained_vgg,
        device=args.device,
    )
    train_model(config)


if __name__ == "__main__":
    main()

