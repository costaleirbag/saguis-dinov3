#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DinoV3HFExtractor:
    """
    Extrai embeddings de imagem usando um modelo DINOv3 do Hugging Face Hub.

    Esta classe gerencia o carregamento do modelo e do processador de imagem,
    e fornece um método para extrair features em lote de forma eficiente.
    """
    def __init__(self,
                 model_name: str,
                 device: str | None = None):
        """
        Inicializa o extrator.

        Args:
            model_name (str): O nome do modelo no Hugging Face Hub.
                Ex: "facebook/dinov3-vitb16-pretrain-lvd1689m"
            device (str, optional): O dispositivo para rodar o modelo ("cuda", "cpu", "mps").
                Se None, tentará usar "mps" ou "cuda" se disponíveis, senão "cpu".
        """
        self.model_name = model_name
        
        # Define o dispositivo, com detecção automática se não for especificado
        if device:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        print(f"[DinoV3HFExtractor] Usando dispositivo: {self.device}")

        # Carrega o processador de imagem e o modelo do Hugging Face Hub
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.inference_mode()
    def embed_pils_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Gera embeddings para um lote de imagens PIL.

        Args:
            images (List[Image.Image]): Uma lista de objetos de imagem PIL.

        Returns:
            List[np.ndarray]: Uma lista de embeddings NumPy, um para cada imagem.
        """
        if not images:
            return []

        # O processador do Hugging Face cuida de toda a transformação:
        # redimensionamento, conversão para tensor e normalização.
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        # O autocast é útil para GPUs que suportam mixed precision (fp16)
        # para acelerar a inferência.
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        # O "pooler_output" contém o embedding agregado da imagem [batch_size, hidden_size]
        embeddings = outputs.pooler_output
        
        # Move para a CPU e converte para uma lista de arrays numpy
        return [emb.cpu().numpy() for emb in embeddings]