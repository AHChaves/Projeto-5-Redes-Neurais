from __future__ import annotations

import argparse
from pathlib import Path

from projeto_redes_neurais.pipeline import (
    run_classical_step,
    run_embedding_step,
    run_visualization_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa a etapa 1 do projeto de embeddings.")
    parser.add_argument("--dataset-root", required=True, help="Raiz local da base CelebA-Spoof.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "embeddings", "visualization", "classical"],
        help="Qual etapa deseja executar.",
    )
    parser.add_argument(
        "--embedding-kind",
        default="deep",
        choices=["deep", "hog"],
        help="Embedding usado nas etapas visualization/classical.",
    )
    parser.add_argument("--viz-method", default="pca", choices=["pca", "tsne", "umap"])
    parser.add_argument("--classical-model", default="svm", choices=["svm", "knn", "logreg"])
    return parser


def resolve_embedding_path(dataset_root: str, split: str, max_samples: int | None, embedding_kind: str) -> Path:
    outputs = run_embedding_step(dataset_root, split=split, max_samples=max_samples)
    return outputs[embedding_kind]


def main() -> None:
    args = build_parser().parse_args()
    target_embedding: Path | None = None

    if args.stage in {"all", "embeddings"}:
        target_embedding = resolve_embedding_path(
            args.dataset_root,
            split=args.split,
            max_samples=args.max_samples,
            embedding_kind=args.embedding_kind,
        )
        print(f"Embeddings gerados em: {target_embedding}")

    if args.stage in {"all", "visualization"}:
        if target_embedding is None:
            target_embedding = resolve_embedding_path(
                args.dataset_root,
                split=args.split,
                max_samples=args.max_samples,
                embedding_kind=args.embedding_kind,
            )
        figure_path = run_visualization_step(target_embedding, method=args.viz_method)
        print(f"Figura salva em: {figure_path}")

    if args.stage in {"all", "classical"}:
        if target_embedding is None:
            target_embedding = resolve_embedding_path(
                args.dataset_root,
                split=args.split,
                max_samples=args.max_samples,
                embedding_kind=args.embedding_kind,
            )
        result = run_classical_step(target_embedding, model_name=args.classical_model)
        print(f"Modelo: {result.model_name}")
        print(f"Acuracia: {result.accuracy:.4f}")
        print(f"Macro F1: {result.macro_f1:.4f}")
        print(result.report)


if __name__ == "__main__":
    main()
