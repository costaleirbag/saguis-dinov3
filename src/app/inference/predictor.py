# src/app/inference/predictor.py (VERSÃO FINAL com pipeline de PCA)
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from joblib import load

# Imports do seu projeto
from app.vision.dinov3_extractor import DinoV3HFExtractor
from app.features.tabular import engineer_tab_features, load_ibge_urban_areas
from app.data.images import load_pil_from_url

@dataclass
class PredictorConfig:
    """Configuração para o predictor final, que carrega um ficheiro de artefactos."""
    artifacts_path: str | Path
    hf_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device_prefer: str = "mps"

class SaguiPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self._load_runtime()

    def _load_runtime(self):
        # Carrega o DINOv3 extractor para gerar embeddings
        self.extractor = DinoV3HFExtractor(
            model_name=self.cfg.hf_model,
            device=self.cfg.device_prefer,
        )

        # Carrega o pacote de artefactos treinado
        artifacts = load(self.cfg.artifacts_path)
        self.embedding_pipeline = artifacts["embedding_pipeline"]
        self.tabular_pipeline = artifacts["tabular_pipeline"]
        self.model = artifacts["model"]
        self.best_threshold = artifacts["best_threshold"]
        self.tabular_feature_names = artifacts["feature_names"]["tabular"]
        
        # Garante que 'tabular_config' exista e contenha os parâmetros urbanos.
        # Isso é crucial para engineer_tab_features, mesmo que os artefatos
        # não tenham essas chaves por padrão.
        self.tabular_config = artifacts.get("tabular_config", {})
        self.tabular_config.setdefault("urban_areas_path", "data/geo/ibge_areas_urbanizadas.gpkg")
        self.tabular_config.setdefault("urban_layer", "lml_area_densamente_edificada_a")
        self.tabular_config.setdefault("urban_radius_km", 5.0)


        print(f"Predictor carregado com sucesso a partir de {self.cfg.artifacts_path}")
        print(f"Limiar de decisão (threshold): {self.best_threshold:.4f}")

    def _embed_pil(self, pil_img: Image.Image) -> np.ndarray:
        """Gera um embedding 1D a partir de uma imagem PIL."""
        return self.extractor.embed_pils_batch([pil_img])[0]

    def predict(
        self,
        image: Image.Image | str,
        observed_on: str,
        latitude: float,
        longitude: float,
    ) -> Dict[str, Any]:
        """
        Prevê a classe de uma observação a partir de uma imagem e dados tabulares.
        A função engineer_tab_features já utiliza a tabular_config para incluir
        features de proximidade urbana, se configurado.
        """
        # 1. Pré-processamento da Imagem
        pil_image = load_pil_from_url(image) if isinstance(image, str) else image
        embedding_vec = self._embed_pil(pil_image).reshape(1, -1)

        # 2. Pré-processamento dos Dados Tabulares
        input_data = pd.DataFrame([{
            "observed_on": observed_on,
            "latitude": latitude,
            "longitude": longitude,
        }])
        
        # A função engineer_tab_features já usa os parâmetros de urban_areas_path,
        # urban_radius_km, e urban_layer contidos em self.tabular_config
        tab_features_df, _ = engineer_tab_features(
            input_data, 
            mode="latlon_time", # Manter o mode para gerar as features tabulares de tempo/localização
            urban_areas_path=self.tabular_config["urban_areas_path"],
            urban_radius_km=self.tabular_config["urban_radius_km"],
            urban_layer=self.tabular_config["urban_layer"],
        )
        
        # Garante a mesma ordem de colunas do treino
        # IMPORTANTE: A lista tabular_feature_names deve conter as novas features urbanas.
        # Caso contrário, o 'transform' do tabular_pipeline pode falhar se as colunas esperadas
        # pelo pipeline de treino não corresponderem às geradas aqui.
        tab_features_df = tab_features_df[self.tabular_feature_names]

        # 3. Aplica os Pipelines Treinados
        embedding_processed = self.embedding_pipeline.transform(embedding_vec)
        tabular_processed = self.tabular_pipeline.transform(tab_features_df)
        
        # 4. Combina para formar o vetor de features final
        final_features = np.hstack([embedding_processed, tabular_processed])

        # 5. Faz a Predição
        prob_h = self.model.predict_proba(final_features)[0, 1]
        label = "H" if prob_h >= self.best_threshold else "N-H"
        
        # As linhas abaixo foram removidas pois eram redundantes e causavam o TypeError.
        # A geração de features urbanas já é tratada pela chamada a engineer_tab_features acima.
        # urban_gdf = load_ibge_urban_areas(path="data/geo/ibge_areas_urbanizadas.gpkg", layer="lml_area_densamente_edificada_a")
        # df_tab, tab_cols = engineer_tab_features(input_data, mode="latlon_time", urban_gdf=urban_gdf)
        
        return {
            "prob_H": float(prob_h),
            "label": label,
            "threshold": self.best_threshold,
        }