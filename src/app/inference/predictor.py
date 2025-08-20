# src/app/inference/predictor.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from joblib import load
import requests
import torch

# Importa a nova classe que usa Hugging Face
from app.vision.dinov3_extractor import DinoV3HFExtractor
from app.features.tabular import engineer_tab_features
from app.data.images import load_pil_from_url

def _parse_date_str(s: str) -> str:
    """
    Normaliza a data para o formato que já usamos no dataset (dd/mm/YYYY).
    Aceita 'YYYY-mm-dd' também, convertendo para dd/mm/YYYY.
    """
    s = str(s).strip()
    if "/" in s and len(s.split("/")[0]) <= 2:
        return s
    try:
        from datetime import datetime
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return s

@dataclass
class PredictorConfig:
    """Configuração atualizada para usar o modelo do Hugging Face."""
    model_path: str                          # Caminho para best_model.joblib
    hf_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m" # Nome do modelo no Hub
    device_prefer: str = "mps"               # "mps" no Mac, "cuda" para Nvidia, ou "cpu"

class SaguiPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self._device = self._pick_device(cfg.device_prefer)
        self._load_runtime()

    def _pick_device(self, prefer: str) -> str:
        if prefer == "mps" and torch.backends.mps.is_available():
            return "mps"
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_runtime(self):
        # Carrega o modelo de classificação (ex: LightGBM, a partir do joblib)
        pack = load(self.cfg.model_path)
        self.model = pack["model"]
        self.scaler = pack.get("scaler", None)
        self.pca = pack.get("pca", None)
        self.used_cols = pack.get("used_cols", None)
        self.tab_mode = pack.get("tab_mode", "latlon_time")
        self.threshold = float(pack.get("threshold", 0.5))

        # --- Ponto principal da mudança ---
        # Instancia o extrator DINOv3 usando o nome do modelo do Hugging Face
        self.extractor = DinoV3HFExtractor(
            model_name=self.cfg.hf_model,
            device=self._device,
        )
        # ------------------------------------

        self._emb_is_pca = self.pca is not None
        self._emb_col_prefix = "pca" if self._emb_is_pca else "emb"

    def _pil_from_url_or_pil(self, image: Image.Image | str) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return load_pil_from_url(image).convert("RGB")

    def _embed_pil(self, pil_img: Image.Image) -> np.ndarray:
        # Retorna um vetor 1D (float32)
        # A nova classe processa a lista de imagens diretamente
        feats = self.extractor.embed_pils_batch([pil_img])
        return feats[0].astype(np.float32)

    def _build_feature_row(
        self,
        emb_vec: np.ndarray,
        observed_on: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> pd.DataFrame:
        row = {
            "observed_on": _parse_date_str(observed_on) if observed_on else None,
            "latitude": latitude,
            "longitude": longitude,
        }
        df = pd.DataFrame([row])
        tab_df, tab_cols = engineer_tab_features(df, mode=self.tab_mode)

        if self.pca is not None:
            emb_vec = self.pca.transform(emb_vec.reshape(1, -1))[0]
        
        emb_cols = [f"{self._emb_col_prefix}_{i}" for i in range(len(emb_vec))]
        emb_df = pd.DataFrame([emb_vec], columns=emb_cols)

        X = pd.concat([emb_df, tab_df], axis=1)

        if self.used_cols:
            for c in self.used_cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[self.used_cols]
        return X

    def predict(
        self,
        image: Image.Image | str,
        observed_on: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Prevê a classe de uma observação a partir de uma imagem e dados tabulares.

        Args:
            image (Image.Image | str): URL da imagem ou um objeto PIL.Image.
            observed_on (str, optional): Data da observação no formato "dd/mm/YYYY" ou "YYYY-mm-dd".
            latitude (float, optional): Latitude da observação.
            longitude (float, optional): Longitude da observação.
            return_intermediate (bool): Se True, retorna dados intermediários para depuração.

        Returns:
            Dict[str, Any]: Um dicionário com a probabilidade, o rótulo e o limiar.
        """
        pil = self._pil_from_url_or_pil(image)
        emb = self._embed_pil(pil)
        X = self._build_feature_row(emb, observed_on, latitude, longitude).astype(np.float32)

        if self.scaler is not None:
            X_np = self.scaler.transform(X.values)
        else:
            X_np = X.values

        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X_np)[0, 1])
        else:
            from scipy.special import expit
            prob = float(expit(self.model.decision_function(X_np))[0])

        label = "H" if prob >= self.threshold else "N-H"

        out = {
            "prob_H": prob,
            "label": label,
            "threshold": self.threshold,
        }
        if return_intermediate:
            out.update({
                "used_cols": self.used_cols,
                "features_row": X.iloc[0].to_dict(),
                "tab_mode": self.tab_mode,
                "emb_dim": len(emb),
                "pca_applied": self._emb_is_pca,
            })
        return out