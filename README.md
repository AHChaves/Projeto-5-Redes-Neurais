# Projeto 5 - Redes Neurais

Base inicial do projeto da disciplina, organizada a partir do roteiro da **Etapa 1 - Estudo e Comparacao de Embeddings**.

O foco inicial do repositório e a base **CelebA-Spoof**, tratando o problema de **face anti-spoofing** com duas representacoes:

1. Embeddings com deep learning usando `PyTorch` e `ResNet50`.
2. Embeddings tradicionais usando `HOG`.

Depois disso, o projeto tambem prepara:

1. Visualizacao com `PCA`, `t-SNE` e `UMAP`.
2. Uso dos embeddings em modelos classicos como `SVM`, `k-NN` e `Logistic Regression`.

## Estrutura

```text
.
├── notebooks/
│   ├── 00_visao_geral_etapa1.ipynb
│   ├── 01_celeba_spoof_preparacao.ipynb
│   ├── 02_embeddings_deep.ipynb
│   └── 03_embeddings_tradicionais_e_avaliacao.ipynb
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   └── embeddings/
├── reports/
│   └── figures/
├── scripts/
│   └── run_etapa1.py
├── src/
│   └── projeto_redes_neurais/
│       ├── config.py
│       ├── pipeline.py
│       ├── data/
│       ├── embeddings/
│       ├── evaluation/
│       └── models/
└── pyproject.toml
```

## Etapas do PDF mapeadas no projeto

### 1. Geracao de embeddings com deep learning

- Arquitetura inicial: `ResNet50` pre-treinada.
- Camada de embedding: saida da penultima camada apos substituir `fc` por `nn.Identity()`.
- Saida: arquivo `.parquet` com colunas de metadados e features.

### 2. Geracao de embeddings tradicionais

- Baseline inicial: `HOG` para imagens.
- Saida: arquivo `.parquet` no mesmo formato, facilitando comparacoes.

### 3. Avaliacao e visualizacao

- Reducao para 2D com `PCA`, `t-SNE` ou `UMAP`.
- Gera figuras em `reports/figures/`.

### 4. Aplicacao final com metodo nao baseado em deep learning

- Classificacao sobre os embeddings usando `SVM`, `k-NN` ou `Logistic Regression`.

## Dataset inicial: CelebA-Spoof

Repositorio de referencia: <https://github.com/ZhangYuanhan-AI/CelebA-Spoof>

O projeto assume que a base sera baixada manualmente e armazenada localmente. O carregador foi preparado para procurar manifestos em um destes caminhos:

- `metas/intra_test/train_label.json`
- `metas/intra_test/test_label.json`
- ou arquivos JSON equivalentes na raiz do dataset

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Exemplo de uso

Executar todas as etapas em um subconjunto:

```bash
python scripts/run_etapa1.py \
  --dataset-root /caminho/para/CelebA-Spoof \
  --split train \
  --max-samples 500 \
  --stage all \
  --embedding-kind deep \
  --viz-method pca \
  --classical-model svm
```

## Fluxo em notebook

Se preferirmos trabalhar de forma mais interativa, o projeto agora tambem possui notebooks:

1. `notebooks/00_visao_geral_etapa1.ipynb`
2. `notebooks/01_celeba_spoof_preparacao.ipynb`
3. `notebooks/02_embeddings_deep.ipynb`
4. `notebooks/03_embeddings_tradicionais_e_avaliacao.ipynb`

Os notebooks usam os modulos em `src/`, o que ajuda a manter o projeto organizado mesmo com uma interface mais exploratoria.

## Proximos passos sugeridos

1. Confirmar a estrutura final do dataset apos o download local.
2. Ajustar o parser dos JSONs da CelebA-Spoof se necessario.
3. Adicionar experimentos comparando `deep` vs `hog` com metricas consistentes.
4. Registrar resultados em tabelas e figuras para o relatorio da disciplina.
