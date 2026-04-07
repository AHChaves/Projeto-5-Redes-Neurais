#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASET_SOURCE = {
    "name": "celeba-spoof",
    "folder_url": "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=drive_link",
    "notes": (
        "Fonte oficial da CelebA-Spoof usada no projeto. "
        "Essa pasta pode conter arquivos zipados e, em alguns casos, partes como .zip.001."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa a base CelebA-Spoof a partir do Google Drive oficial."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Pasta onde o download sera salvo.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="So mostra o link oficial e a pasta de destino, sem baixar.",
    )
    return parser.parse_args()


def ensure_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "A biblioteca 'gdown' nao esta instalada.\n"
            "Instale com: pip install gdown"
        ) from exc
    return gdown


def list_downloaded_files(target_dir: Path) -> None:
    files = sorted(path for path in target_dir.rglob("*") if path.is_file())
    if not files:
        print("Nenhum arquivo encontrado apos o download.")
        return

    print("\nArquivos encontrados:")
    for path in files[:20]:
        print("-", path)
    if len(files) > 20:
        print(f"... e mais {len(files) - 20} arquivos")

    multipart = [path for path in files if path.name.endswith(".zip.001") or path.name.endswith(".7z.001")]
    if multipart:
        print("\nAviso:")
        print("Foram encontrados arquivos multipartidos, por exemplo:")
        for path in multipart[:5]:
            print("-", path.name)
        print("Para extrair esse tipo de arquivo, use 7-Zip/7z com todas as partes presentes.")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    target_dir = output_root / DATASET_SOURCE["name"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {DATASET_SOURCE['name']}")
    print(f"Destino: {target_dir}")
    print(f"Link oficial: {DATASET_SOURCE['folder_url']}")
    print(f"Observacao: {DATASET_SOURCE['notes']}")

    if args.skip_download:
        return 0

    gdown = ensure_gdown()

    print("\nIniciando download da pasta do Google Drive...")
    print("Isso pode levar bastante tempo, dependendo do tamanho da base e da conexao.")

    downloaded = gdown.download_folder(
        url=DATASET_SOURCE["folder_url"],
        output=str(target_dir),
        quiet=False,
        remaining_ok=True,
        use_cookies=False,
    )

    if not downloaded:
        print("\nNenhum arquivo foi baixado.")
        print("Possiveis causas:")
        print("- limite do Google Drive")
        print("- necessidade de aceitar aviso manualmente no navegador")
        print("- pasta com permissao/restricao temporaria")
        print("\nTente abrir o link oficial no navegador e baixar manualmente, se necessario.")
        return 1

    print(f"\nDownload concluido. Total de itens baixados: {len(downloaded)}")
    list_downloaded_files(target_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
