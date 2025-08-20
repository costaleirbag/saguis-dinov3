# Classificador de Saguis com DINOv3

Este projeto implementa um classificador multimodal para distinguir entre espécies de saguis (ex: Híbrido vs. Não-Híbrido) a partir de imagens de observação. O modelo utiliza features de imagem de última geração extraídas pela arquitetura **DINOv3** da Meta AI, combinadas com dados tabulares contextuais (geolocalização e data) para realizar a predição.

A aplicação conta com uma interface web interativa construída com Streamlit para demonstração e predições individuais.

##  Demo

![Demonstração do App Streamlit](demo_dinov3_saguis.gif)

## Principais Funcionalidades

- **Extração de Features de Imagem**: Utiliza o modelo DINOv3 pré-treinado (via Hugging Face `transformers`) para gerar embeddings de alta qualidade a partir das imagens.
- **Modelo Multimodal**: Combina os embeddings da imagem com features de engenharia de dados tabulares (latitude, longitude, e data da observação).
- **Classificação**: Um modelo tradicional de Machine Learning (ex: XGBoost/LightGBM) é treinado sobre os dados combinados para a classificação final.
- **Interface Web**: Uma aplicação interativa com Streamlit que permite ao usuário fornecer uma URL de imagem, data e localização no mapa para obter uma predição em tempo real.
- **Interface de Linha de Comando (CLI)**: Scripts para realizar a extração de features em lote e para fazer predições individuais via terminal.
- **Gerenciamento de Ambiente**: Utiliza **Poetry** para gerenciamento de dependências e **pyenv** para garantir a versão correta do Python, tornando o projeto robusto e reprodutível.

## Arquitetura e Tecnologias Utilizadas

- **Linguagem**: Python 3.12+
- **Gestão de Ambiente**: Poetry & Pyenv
- **Deep Learning**: PyTorch & Hugging Face Transformers (para DINOv3)
- **Machine Learning**: Scikit-learn, XGBoost/LightGBM
- **Manipulação de Dados**: Pandas, NumPy
- **Interface Web**: Streamlit & Folium
- **Versionamento**: Git & GitHub

## Estrutura do Projeto

```
saguis-dinov3/
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── README.md
└── src/
    └── app/
        ├── cli/
        │   └── predict_one.py
        ├── data/
        │   └── images.py
        ├── features/
        │   └── tabular.py
        ├── inference/
        │   └── predictor.py
        ├── pipeline/
        │   └── make_embeddings.py
        ├── ui/
        │   └── streamlit_app.py
        └── vision/
            └── dinov3_extractor.py
```

## Instalação e Configuração

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

## Como Usar o Projeto

### 1. Rodar a Aplicação Web (Streamlit)

Para iniciar a interface interativa, execute:
```bash
poetry run streamlit run src/app/ui/streamlit_app.py
```
Acesse a URL local (geralmente `http://localhost:8501`) no seu navegador.

### 2. Fazer uma Predição pela Linha de Comando (CLI)

Use o script `predict_one.py` para fazer uma predição única.
```bash
poetry run python src/app/cli/predict_one.py \
    --model "caminho/para/best_model.joblib" \
    --hf_model "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --img_url "URL_DA_IMAGEM" \
    --date "20/08/2025" \
    --lat -23.55 \
    --lon -46.63
```

### 3. Gerar Embeddings em Lote

Para processar um conjunto de dados (CSV) e extrair os embeddings DINOv3, use o script `make_embeddings.py`.
```bash
poetry run python src/app/pipeline/make_embeddings.py \
    --csv_glob "caminho/para/*.csv" \
    --hf_model "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --out "caminho/para/embeddings.parquet"
```

## Próximos Passos e Melhorias

- [ ] Realizar fine-tuning do DINOv3 em vez de apenas usar como extrator de features.
- [ ] Experimentar com outros backbones da família DINOv3 (ViT-Large, etc.).
- [ ] Empacotar o modelo e a aplicação com Docker para facilitar o deploy.
- [ ] Fazer o deploy da aplicação Streamlit em uma plataforma cloud (ex: Streamlit Community Cloud, Hugging Face Spaces).
- [ ] Adicionar um fluxo de treinamento completo e versionamento de modelos (ex: MLflow).

## Licença

Este projeto está sob a licença [MIT](LICENSE).

## Contato

[Seu Nome] - [seu-email@exemplo.com] - [Link para seu LinkedIn/GitHub]