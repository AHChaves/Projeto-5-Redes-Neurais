from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from projeto_redes_neurais.data.celebA_spoof import load_manifest


def extract_hog_embeddings(
    dataset_root: str | Path,
    split: str,
    output_path: str | Path,
    max_samples: int | None = None,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    manifest = load_manifest(dataset_root, split=split, max_samples=max_samples)

    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        image = Image.open(dataset_root / row.image_path).convert("RGB").resize((224, 224))
        gray = rgb2gray(image)
        embedding = hog(
            gray,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        sample = {"image_path": row.image_path, "label": int(row.label)}
        for index, value in enumerate(embedding):
            sample[f"f_{index:04d}"] = float(value)
        rows.append(sample)

    frame = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame
