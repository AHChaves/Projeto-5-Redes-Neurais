from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SPLIT_FILES = {
    "train": "train_label.json",
    "validation": "test_label.json",
    "test": "test_label.json",
}


@dataclass(slots=True)
class CelebASample:
    image_path: str
    label: int
    subject_id: str | None = None


def _normalize_label(raw_label: object) -> int:
    if isinstance(raw_label, list) and raw_label:
        return int(raw_label[-1])
    return int(raw_label)


def load_manifest(dataset_root: str | Path, split: str = "train", max_samples: int | None = None) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    split_file = SPLIT_FILES.get(split, f"{split}.json")
    manifest_path = dataset_root / "metas" / "intra_test" / split_file
    if not manifest_path.exists():
        manifest_path = dataset_root / split_file
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifesto da CelebA-Spoof nao encontrado. "
            "Esperado em 'metas/intra_test/<split>_label.json' ou na raiz do dataset."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows: list[dict[str, object]] = []
    for image_path, raw_metadata in data.items():
        if isinstance(raw_metadata, dict):
            label = raw_metadata.get("live_spoof", raw_metadata.get("label", 0))
            subject_id = raw_metadata.get("subject_id")
        else:
            label = raw_metadata
            subject_id = None

        rows.append(
            {
                "image_path": image_path,
                "label": _normalize_label(label),
                "subject_id": subject_id,
                "split": split,
            }
        )

    frame = pd.DataFrame(rows)
    if max_samples:
        frame = frame.head(max_samples).copy()
    return frame


class CelebASpoofDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples = load_manifest(self.dataset_root, split=split, max_samples=max_samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples.iloc[index]
        image_path = self.dataset_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "label": int(row["label"]),
            "image_path": str(row["image_path"]),
        }
