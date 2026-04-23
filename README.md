# Projeto 5 - Redes Neurais

Este repositorio foi reorganizado para trabalhar apenas com notebooks `.ipynb`.

O foco da Etapa 1 agora esta na base **CelebA Align** localizada em `data/raw/img_align_celeba`, comparando:

1. Embeddings com deep learning usando `PyTorch` e `ResNet50`.
2. Embeddings tradicionais usando `HOG`.
3. Geracao complementar de imagens sinteticas com `StyleGAN2`.

Depois disso, os notebooks tambem fazem:

1. Visualizacao com `PCA`, `t-SNE` e `UMAP`.
2. Classificacao com metodos classicos como `SVM`, `k-NN` e `Logistic Regression`.

## Estrutura

```text
.
├── notebooks/
│   ├── 00_visao_geral_etapa1.ipynb
│   ├── 01_celeba_preparacao.ipynb
│   ├── 02_geracao_sintetica_stylegan2.ipynb
│   ├── 03_embeddings_deep.ipynb
│   └── 04_embeddings_tradicionais_e_avaliacao.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── stylegan2_runs/
│   └── embeddings/
└── reports/
    └── figures/
```

## Ordem sugerida

1. Abrir `notebooks/01_celeba_preparacao.ipynb`
2. Executar `notebooks/02_geracao_sintetica_stylegan2.ipynb`
3. Executar `notebooks/03_embeddings_deep.ipynb`
4. Executar `notebooks/04_embeddings_tradicionais_e_avaliacao.ipynb`

## Base usada

Diretorio local utilizado no projeto:

`/home/arthur/Documentos/Github/Projeto-5-Redes-Neurais/data/raw/img_align_celeba`

Como a pasta atual contem apenas as imagens alinhadas, os notebooks trabalham com embeddings e clustering nao supervisionado. A etapa de geracao sintetica agora usa StyleGAN2-ADA em `256x256`.
