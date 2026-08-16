from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .report import write_json
from .vla_dataset import VLAPretrainDataset, collate_vla_samples


def _token_ids(texts: List[str], vocabulary_size: int = 32768, length: int = 24) -> np.ndarray:
    rows = []
    for text in texts:
        tokens = re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE)[:length]
        values = [
            1 + int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % (vocabulary_size - 1)
            for token in tokens
        ]
        rows.append((values + [0] * length)[:length])
    return np.asarray(rows, dtype=np.int64)


def smoke_vla_train(
    manifest: Path,
    output: Path,
    *,
    split: str = "train",
    batch_size: int = 2,
    steps: int = 5,
    learning_rate: float = 3e-4,
    allow_technical_candidates: bool = False,
    device: str = "cuda",
    seed: int = 0,
) -> Dict[str, Any]:
    if batch_size < 2:
        raise ValueError("对比学习 smoke test 的 batch_size 必须 >= 2")
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("smoke-vla-train 需要安装 torch（pip install -e '.[vla]'）") from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 未检测到 GPU")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = VLAPretrainDataset(
        manifest,
        split=split,
        allow_technical_candidates=allow_technical_candidates,
        seed=seed,
    )
    if len(dataset) < batch_size:
        raise ValueError(f"当前 split 只有 {len(dataset)} 条，少于 batch_size={batch_size}")
    samples = [dataset[index] for index in range(batch_size)]
    batch = collate_vla_samples(samples)
    frames = torch.from_numpy(batch["frames"]).to(device=device, dtype=torch.float32)
    frames = frames.permute(0, 1, 4, 2, 3).contiguous().div_(255.0)
    # Smoke model uses eight frames per clip to verify learning while keeping the run cheap.
    frames = frames[:, ::max(1, frames.shape[1] // 8)][:, :8]
    tokens = torch.from_numpy(_token_ids(batch["texts"])).to(device=device)
    masks = {
        name: torch.from_numpy(value).to(device=device)
        for name, value in batch["loss_masks"].items()
    }

    class TinyVLAPretrainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.temporal_predictor = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
            self.text_embedding = nn.Embedding(32768, 64, padding_idx=0)
            self.log_temperature = nn.Parameter(torch.tensor(-2.3))

        def encode_video(self, video):
            batch_n, time_n = video.shape[:2]
            features = self.visual(video.reshape(batch_n * time_n, *video.shape[2:])).flatten(1)
            return features.reshape(batch_n, time_n, -1)

        def encode_text(self, token_batch):
            valid = token_batch.ne(0).unsqueeze(-1)
            embedded = self.text_embedding(token_batch)
            return (embedded * valid).sum(1) / valid.sum(1).clamp_min(1)

    model = TinyVLAPretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    use_bfloat16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    def contrastive(left, right, mask):
        indices = torch.nonzero(mask > 0.5, as_tuple=False).flatten()
        if indices.numel() < 2:
            return left.sum() * 0.0
        left = functional.normalize(left[indices], dim=-1)
        right = functional.normalize(right[indices], dim=-1)
        temperature = model.log_temperature.exp().clamp(0.01, 1.0)
        logits = left @ right.T / temperature
        labels = torch.arange(indices.numel(), device=left.device)
        return 0.5 * (
            functional.cross_entropy(logits, labels)
            + functional.cross_entropy(logits.T, labels)
        )

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        brightness = 0.85 + 0.3 * torch.rand((frames.shape[0], 1, 1, 1, 1), device=device)
        augmented = torch.flip(frames * brightness, dims=[4]).clamp(0, 1)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            sequence = model.encode_video(frames)
            augmented_sequence = model.encode_video(augmented)
            video_loss = contrastive(
                sequence.mean(1), augmented_sequence.mean(1), masks["video_representation"]
            )
            midpoint = max(1, sequence.shape[1] // 2)
            predicted_future = model.temporal_predictor(sequence[:, :midpoint].mean(1))
            future_target = sequence[:, midpoint:].mean(1).detach()
            temporal_per_sample = functional.mse_loss(
                predicted_future, future_target, reduction="none"
            ).mean(1)
            temporal_mask = masks["temporal_prediction"]
            temporal_loss = (temporal_per_sample * temporal_mask).sum() / temporal_mask.sum().clamp_min(1)
            text_features = model.encode_text(tokens)
            text_loss = contrastive(
                sequence.mean(1), text_features, masks["video_text_alignment"]
            )
            total = video_loss + temporal_loss + text_loss
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append({
            "step": step + 1,
            "total_loss": float(total.detach().cpu()),
            "video_contrastive_loss": float(video_loss.detach().cpu()),
            "temporal_prediction_loss": float(temporal_loss.detach().cpu()),
            "video_text_loss": float(text_loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        })

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "vla-smoke-checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "manifest": str(manifest.expanduser().resolve()),
        "technical_candidate_mode": allow_technical_candidates,
    }, checkpoint)
    report = {
        "status": "succeeded",
        "purpose": "pipeline smoke test, not a production pretrained checkpoint",
        "manifest": str(manifest.expanduser().resolve()),
        "split": split,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "bfloat16": use_bfloat16,
        "batch_size": batch_size,
        "input_shape": list(frames.shape),
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_second": steps / elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "objectives_trained": ["video_representation", "temporal_prediction", "video_text_alignment"],
        "objectives_not_present": ["mano_motion", "robot_action", "camera_pose", "tactile"],
        "technical_candidate_mode": allow_technical_candidates,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    if device.startswith("cuda"):
        report["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    write_json(output / "vla-train-smoke.json", report)
    return report
