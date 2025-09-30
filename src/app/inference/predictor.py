# src/app/inference/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from joblib import load

from app.vision.dinov3_extractor import DinoV3HFExtractor
from app.features.tabular import engineer_tab_features
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
        # Extrator de embeddings
        self.extractor = DinoV3HFExtractor(
            model_name=self.cfg.hf_model,
            device=self.cfg.device_prefer,
        )

        # Artefatos de treino
        artifacts = load(self.cfg.artifacts_path)
        self.embedding_pipeline = artifacts["embedding_pipeline"]
        self.tabular_pipeline = artifacts["tabular_pipeline"]
        self.model = artifacts["model"]
        self.best_threshold = artifacts["best_threshold"]
        self.tabular_feature_names = artifacts["feature_names"]["tabular"]

        # Config tabular (IBGE) — garantimos defaults
        self.tabular_config = artifacts.get("tabular_config", {}) or {}
        self.tabular_config.setdefault("urban_areas_path", "data/geo/ibge_areas_urbanizadas.gpkg")
        self.tabular_config.setdefault("urban_layer", "lml_area_densamente_edificada_a")
        self.tabular_config.setdefault("urban_radius_km", 5.0)

        # (Opcional) assinatura de pré-processamento para auditoria
        self.preprocess_signature = {
            "hf_model": self.cfg.hf_model,
            "embedding_pipeline": type(self.embedding_pipeline).__name__,
            "tabular_pipeline": type(self.tabular_pipeline).__name__,
            "model": type(self.model).__name__,
        }

        print(f"[Predictor] Carregado de: {self.cfg.artifacts_path}")
        print(f"[Predictor] Threshold: {self.best_threshold:.4f}")
        print(f"[Predictor] Preprocess: {self.preprocess_signature}")

    # ---------- Helpers internos ----------
    def _embed_pil(self, pil_img: Image.Image) -> np.ndarray:
        """Gera um embedding 1D a partir de uma imagem PIL."""
        return self.extractor.embed_pils_batch([pil_img])[0]

    def _build_tabular_df(self, observed_on: str, latitude: float, longitude: float) -> pd.DataFrame:
        input_data = pd.DataFrame([{
            "observed_on": observed_on,
            "latitude": latitude,
            "longitude": longitude,
        }])
        tab_df, _ = engineer_tab_features(
            input_data,
            mode="latlon_time",
            urban_areas_path=self.tabular_config["urban_areas_path"],
            urban_radius_km=self.tabular_config["urban_radius_km"],
            urban_layer=self.tabular_config["urban_layer"],
        )
        # ordem idêntica à do treino
        tab_df = tab_df[self.tabular_feature_names]
        return tab_df

    def _predict_from_vectors(self, emb_vec_2d: np.ndarray, tab_df: pd.DataFrame) -> Dict[str, Any]:
        # pipelines
        emb_proc = self.embedding_pipeline.transform(emb_vec_2d)
        tab_proc = self.tabular_pipeline.transform(tab_df)
        X = np.hstack([emb_proc, tab_proc])

        prob_h = float(self.model.predict_proba(X)[0, 1])
        label = "H" if prob_h >= self.best_threshold else "N-H"
        return {"prob_H": prob_h, "label": label, "threshold": self.best_threshold}

    # ---------- API pública ----------
    def predict(
        self,
        image: Image.Image | str,
        observed_on: str,
        latitude: float,
        longitude: float,
        *,
        force_resize_224: bool = False,
    ) -> Dict[str, Any]:
        """
        Prediz a classe a partir de uma imagem (PIL ou URL) + dados tabulares.
        """
        pil_image = load_pil_from_url(image) if isinstance(image, str) else image
        if not isinstance(pil_image, Image.Image):
            raise ValueError("Parâmetro 'image' inválido.")

        if force_resize_224:
            pil_image = pil_image.resize((224, 224), Image.BICUBIC)

        emb_vec = self._embed_pil(pil_image).reshape(1, -1)
        tab_df = self._build_tabular_df(observed_on, latitude, longitude)
        return self._predict_from_vectors(emb_vec, tab_df)

    def predict_from_embedding(
        self,
        embedding_vec: np.ndarray,
        observed_on: str,
        latitude: float,
        longitude: float,
    ) -> Dict[str, Any]:
        """
        Prediz a classe usando um EMBEDDING PRONTO (bypass de imagem/YOLO).
        Útil para paridade com parquet/notebook.
        """
        if embedding_vec.ndim == 1:
            emb = embedding_vec.reshape(1, -1)
        elif embedding_vec.ndim == 2 and embedding_vec.shape[0] == 1:
            emb = embedding_vec
        else:
            raise ValueError("embedding_vec deve ser (D,) ou (1, D).")

        tab_df = self._build_tabular_df(observed_on, latitude, longitude)
        return self._predict_from_vectors(emb, tab_df)
