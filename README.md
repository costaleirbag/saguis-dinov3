# Classificador de Saguis com DINOv3

Classificador multimodal (Imagem + Tabular) para identificar saguis H vs N-H. As imagens são processadas com um recorte automático via YOLO (mesmo recorte do treino), os embeddings vêm do DINOv3, e enriquecemos os dados tabulares com latitude/longitude, data e proximidade a áreas urbanas (IBGE). O pacote inclui HPO (Optuna+MLflow), treino final (com/sem PCA) e app Streamlit.

##  Demo

![Demonstração do App Streamlit](demo_dinov3_saguis.gif)

## Principais funcionalidades

- Imagem (DINOv3): extrai embeddings com o backbone DINOv3 (via Hugging Face), após recorte YOLO replicando o pipeline de treino.
- Tabular (Geo+Tempo): latitude, longitude e data (ano, mês, sen/cos sazonais).
- Proximidade urbana (IBGE): recursos de geoproximidade a polígonos urbanos (dentro/fora, distâncias em metros, contagem em raio km).
- HPO: busca de hiperparâmetros com Optuna e rastreamento no MLflow.
- Treino final: LightGBM com pipelines de pré-processamento, opção de PCA nos embeddings.
- PCA opcional: passe `--pca_components -1` para desativar PCA e usar apenas scaling.
- App Streamlit: demonstração e predições individuais com o mesmo pré-processamento do treino.
- CLI: geração de embeddings em lote e predição única via terminal.
- Artefatos ricos: salvamos pipelines, lista de features, threshold ótimo, métricas, importâncias de features e configuração tabular usada.

## Arquitetura e tecnologias

- Linguagem: Python 3.12+
- Gestão de ambiente: Poetry & Pyenv
- Visão computacional: PyTorch + Transformers (DINOv3)
- Detecção/recorte: Ultralytics YOLO (pré-processamento de imagens)
- Machine Learning: scikit-learn, LightGBM (e XGBoost opcional em outros scripts)
- Geoespacial: GeoPandas, Shapely (proximidade IBGE), opcional Rtree/pygeos para indexação
- HPO/Tracking: Optuna + MLflow
- Web: Streamlit & Folium

## Estrutura do projeto (resumo)

```
src/app/
    cli/
        predict_one.py               # Predição única via terminal
    data/
        images.py                    # IO de imagens (URL -> PIL)
        tabular.py                   # Utilitários de dados tabulares
    features/
        tabular.py                   # Engenharia de features (lat/lon/tempo + IBGE)
    inference/
        predictor.py                 # Pipeline de inferência (usa artefatos .joblib)
    pipeline/
        make_embeddings.py           # Gera embeddings DINOv3 em lote
        preprocess_and_filter_images.py # Recorte YOLO consistente com treino
        hyperparam_search.py         # HPO Optuna + MLflow
        train_final_model.py         # Treino final (sem PCA)
        train_pca_model.py           # Treino final com PCA opcional
    ui/
        streamlit_app.py             # App web
    utils/
        io.py, hash.py, ...
    vision/
        dinov3_extractor.py          # Extrator DINOv3 (Hugging Face)
```

## Instalação e configuração

Siga os passos abaixo para configurar o ambiente e rodar o projeto localmente.

**Pré-requisitos:**
- [Git](https://git-scm.com/)
- [pyenv](https://github.com/pyenv/pyenv) (recomendado para gerenciar a versão do Python)
- [Poetry](https://python-poetry.org/) (para gerenciar as dependências do projeto)

**Passos:**

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/saguis-dinov3.git](https://github.com/seu-usuario/saguis-dinov3.git)
    cd saguis-dinov3
    ```

2.  **Configure a versão correta do Python:**
    Este projeto requer Python `~3.12`. Use o `pyenv` para garantir que você está usando a versão correta.
    ```bash
    # Define a versão do Python para esta pasta
    pyenv local 3.12.11 
    # (Instale com 'pyenv install 3.12.11' se você não a tiver)
    ```

3.  **Instale as dependências com o Poetry:**
    O Poetry irá ler o arquivo `pyproject.toml`, criar um ambiente virtual e instalar todas as bibliotecas necessárias.
    ```bash
    poetry install
    ```

4.  **Autenticação no Hugging Face:**
    O modelo DINOv3 é um "gated model". Você precisa aceitar os termos na página do modelo e fazer login via terminal.
    ```bash
    # Este comando pedirá um token que você pode gerar nas configurações da sua conta Hugging Face
    huggingface-cli login
    ```

Notas geoespaciais:
- Para usar as features urbanas (IBGE), baixe um arquivo local (ex.: `data/geo/ibge_areas_urbanizadas.gpkg`). Se houver várias camadas, informe `--urban_layer` (ex.: `lml_area_densamente_edificada_a`).
- GeoPandas/Shapely já estão listados; para melhor desempenho, instale `rtree`.

## Como usar

### 1. Rodar a Aplicação Web (Streamlit)

Para iniciar a interface interativa, execute:
```bash
poetry run streamlit run src/app/ui/streamlit_app.py
```
Acesse a URL local (geralmente `http://localhost:8501`) no seu navegador.

### 2. Predição via Linha de Comando (CLI)

Use o script `predict_one.py` para fazer uma predição única.
```bash
poetry run python src/app/cli/predict_one.py \
    --model "outputs/final_model/final_model.joblib" \
    --hf_model "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --img_url "URL_DA_IMAGEM" \
    --date "20/08/2025" \
    --lat -23.55 \
    --lon -46.63
```

O predictor carrega os artefatos e usa a mesma configuração tabular salva (incl. IBGE) para garantir consistência com o treino.

### 3. Gerar embeddings em lote

Para processar um conjunto de dados (CSV) e extrair os embeddings DINOv3, use o script `make_embeddings.py`.
```bash
poetry run python src/app/pipeline/make_embeddings.py \
    --csv_glob "caminho/para/*.csv" \
    --hf_model "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --out "caminho/para/embeddings.parquet"
```

Se desejar garantir recorte YOLO consistente antes de gerar embeddings, use/utilize a lógica de `preprocess_and_filter_images.py` no seu fluxo de preparação.

### 4. HPO (Optuna + MLflow)

```bash
poetry run python src/app/pipeline/hyperparam_search.py \
    --embeddings outputs/embeddings/embeddings_yolo_processed.parquet \
    --tab_mode latlon_time \
    --n_trials 100 \
    --out_dir outputs/hpo_run \
    --experiment_name Saguis-Classifier-HPO \
    --urban_areas_path data/geo/ibge_areas_urbanizadas.gpkg \
    --urban_layer lml_area_densamente_edificada_a \
    --urban_radius_km 5
```

O script salva `tabular_config.json` (incluindo IBGE) junto dos resultados, além da lista de features usada.

### 5. Treino final (sem PCA) e com PCA opcional

- Com PCA opcional (desligue com `--pca_components -1`):

```bash
poetry run python src/app/pipeline/train_pca_model.py \
    --embeddings outputs/embeddings/embeddings_yolo_processed.parquet \
    --hpo_dir outputs/hpo_run \
    --pca_components 128 \
    --model_out_dir outputs/final_model_pca
```

Ambos salvam um `.joblib` com pipelines, modelo, threshold ótimo e config tabular. Também salvamos `feature_importances` do LightGBM mapeadas aos nomes das features (incluindo PCA ou não).

### 6. App Streamlit

```bash
poetry run streamlit run src/app/ui/streamlit_app.py
```

O app usa a mesma função de processamento de imagem do pipeline (recorte YOLO) e o mesmo conjunto de features tabulares (incl. IBGE) que foram usados no treino.

## Referências ao DINOv3

O DINOv3 é um backbone de visão computacional desenvolvido pela Meta AI, projetado para tarefas de aprendizado auto-supervisionado em imagens. Ele utiliza a arquitetura Vision Transformer (ViT) e é otimizado para extração de embeddings ricos e generalizáveis. Mais detalhes podem ser encontrados na [página oficial do modelo no Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

No contexto deste projeto, o DINOv3 é usado para gerar embeddings de imagens após o recorte automático via YOLO. Esses embeddings são então combinados com dados tabulares para treinar um classificador multimodal.

## Próximos passos e melhorias

- [ ] Realizar fine-tuning do DINOv3 em vez de apenas usar como extrator de features.
- [ ] Experimentar com outros backbones da família DINOv3 (ViT-Large, etc.).
- [ ] Empacotar o modelo e a aplicação com Docker para facilitar o deploy.
- [ ] Fazer o deploy da aplicação Streamlit em uma plataforma cloud (ex: Streamlit Community Cloud, Hugging Face Spaces).
- [ ] Adicionar um fluxo de treinamento completo e versionamento de modelos (ex: MLflow).

## Artefatos salvos (estrutura)

Os `.joblib` finais incluem, pelo menos:

- embedding_pipeline: Pipeline de embeddings (Scaler [+ PCA, se aplicável])
- tabular_pipeline: Pipeline de tabular (Scaler)
- model: Classificador (LightGBM)
- feature_names: { embeddings_pca: [...], tabular: [...] }
- best_threshold: Limiar sugerido (com base no treino)
- metrics: métricas principais (ex.: AUC)
- feature_importances: lista ordenada de {feature, importance} (se disponível)
- tabular_config: { tab_mode, urban_areas_path, urban_layer, urban_radius_km }

## Licença

Este projeto está sob a licença [MIT](LICENSE).

## Contato

[Seu Nome] - [seu-email@exemplo.com] - [Link para seu LinkedIn/GitHub]