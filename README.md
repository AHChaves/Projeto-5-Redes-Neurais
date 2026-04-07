# Projeto 5 - Redes Neurais

Este repositorio foi reorganizado para trabalhar apenas com notebooks `.ipynb`.

O foco da Etapa 1 continua sendo a base **CelebA-Spoof**, comparando:

1. Embeddings com deep learning usando `PyTorch` e `ResNet50`.
2. Embeddings tradicionais usando `HOG`.

Depois disso, os notebooks tambem fazem:

1. Visualizacao com `PCA`, `t-SNE` e `UMAP`.
2. Classificacao com metodos classicos como `SVM`, `k-NN` e `Logistic Regression`.

## Estrutura

```text
.
├── notebooks/
│   ├── 00_visao_geral_etapa1.ipynb
│   ├── 01_celeba_spoof_preparacao.ipynb
│   ├── 02_embeddings_deep.ipynb
│   └── 03_embeddings_tradicionais_e_avaliacao.ipynb
├── data/
│   └── raw/
├── artifacts/
│   └── embeddings/
└── reports/
    └── figures/
```

## Ordem sugerida

1. Abrir `notebooks/01_celeba_spoof_preparacao.ipynb`
2. Executar `notebooks/02_embeddings_deep.ipynb`
3. Executar `notebooks/03_embeddings_tradicionais_e_avaliacao.ipynb`

## Base usada

Repositorio oficial da CelebA-Spoof:

<https://github.com/ZhangYuanhan-AI/CelebA-Spoof>

O notebook de preparacao explica como organizar o download local e validar os manifestos JSON da base.
