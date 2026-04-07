from __future__ import annotations

from pathlib import Path

import pandas as pd

from projeto_redes_neurais.config import DEFAULT_PATHS
from projeto_redes_neurais.embeddings.deep import extract_deep_embeddings
from projeto_redes_neurais.embeddings.traditional import extract_hog_embeddings
from projeto_redes_neurais.evaluation.visualization import reduce_embeddings, save_scatter_plot
from projeto_redes_neurais.models.classical import TrainingResult, train_classical_model


def run_embedding_step(
    dataset_root: str | Path,
    split: str = "train",
    max_samples: int | None = None,
) -> dict[str, Path]:
    DEFAULT_PATHS.ensure()
    output_dir = DEFAULT_PATHS.artifacts_dir / "embeddings" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    deep_path = output_dir / "deep_resnet50.parquet"
    hog_path = output_dir / "traditional_hog.parquet"
    extract_deep_embeddings(dataset_root=dataset_root, split=split, output_path=deep_path, max_samples=max_samples)
    extract_hog_embeddings(dataset_root=dataset_root, split=split, output_path=hog_path, max_samples=max_samples)
    return {"deep": deep_path, "hog": hog_path}


def run_visualization_step(embedding_path: str | Path, method: str = "pca") -> Path:
    frame = pd.read_parquet(embedding_path)
    reduced = reduce_embeddings(frame, method=method)
    output_path = DEFAULT_PATHS.reports_dir / "figures" / f"{Path(embedding_path).stem}_{method}.png"
    title = f"{Path(embedding_path).stem} - {method.upper()}"
    return save_scatter_plot(reduced, output_path=output_path, title=title)


def run_classical_step(embedding_path: str | Path, model_name: str = "svm") -> TrainingResult:
    frame = pd.read_parquet(embedding_path)
    return train_classical_model(frame, model_name=model_name)
