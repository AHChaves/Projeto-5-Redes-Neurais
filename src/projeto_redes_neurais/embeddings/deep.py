from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from projeto_redes_neurais.data.celebA_spoof import CelebASpoofDataset


def build_resnet50_embedding_model() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model


def extract_deep_embeddings(
    dataset_root: str | Path,
    split: str,
    output_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    dataset = CelebASpoofDataset(dataset_root=dataset_root, split=split, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_embedding_model().to(resolved_device)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(resolved_device)
            embeddings = model(images).cpu().numpy()
            labels = batch["label"].tolist()
            image_paths = batch["image_path"]
            for embedding, label, image_path in zip(embeddings, labels, image_paths, strict=False):
                row = {"image_path": image_path, "label": label}
                for index, value in enumerate(embedding):
                    row[f"f_{index:04d}"] = float(value)
                rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame
