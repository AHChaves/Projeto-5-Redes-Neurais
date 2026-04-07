from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def load_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = frame[["image_path", "label"]].copy()
    features = frame.drop(columns=["image_path", "label"])
    return meta, features


def reduce_embeddings(frame: pd.DataFrame, method: str = "pca", random_state: int = 42) -> pd.DataFrame:
    meta, features = load_feature_matrix(frame)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
    elif method == "umap":
        if umap is None:
            raise ImportError("Instale 'umap-learn' para usar UMAP.")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Metodo desconhecido: {method}")

    reduced = reducer.fit_transform(features)
    result = meta.copy()
    result["x"] = reduced[:, 0]
    result["y"] = reduced[:, 1]
    result["method"] = method
    return result


def save_scatter_plot(frame_2d: pd.DataFrame, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=frame_2d, x="x", y="y", hue="label", alpha=0.7, palette="Set2", s=35)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
