# src/app/pipeline/make_embeddings.py (VERSÃO ATUALIZADA para ficheiros locais)
import argparse
from pathlib import Path
from typing import List
from PIL import Image

import pandas as pd
from tqdm import tqdm

from app.vision.dinov3_extractor import DinoV3HFExtractor
from app.utils.io import save_parquet

def main():
    parser = argparse.ArgumentParser(description="Generate DINOv3 embeddings from a CSV of image paths.")
    # --- Argumentos atualizados ---
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to the metadata CSV file (e.g., processed_metadata.csv)")
    parser.add_argument("--image_path_col", type=str, default="processed_image_path", help="Name of the column containing the local image file paths")
    
    parser.add_argument("--hf_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="DINOv3 model from Hugging Face Hub")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps'). Auto-detects if not provided.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", type=str, required=True, help="Path to save the output embeddings parquet file (e.g., embeddings_yolo.parquet)")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    print(f"Encontrados {len(df)} registos no ficheiro de metadados.")

    # --- Lógica de carregamento de imagens locais ---
    images: List[Image.Image] = []
    valid_indices = []
    pbar_load = tqdm(df.iterrows(), total=len(df), desc="A carregar imagens locais")
    for index, row in pbar_load:
        try:
            img_path = Path(row[args.image_path_col])
            if img_path.exists():
                images.append(Image.open(img_path).convert("RGB"))
                valid_indices.append(index)
            else:
                print(f"Aviso: ficheiro de imagem não encontrado em {img_path}, a ignorar.")
        except Exception:
            # Ignora imagens que não podem ser abertas
            continue
    
    # Filtra o DataFrame para manter apenas as linhas correspondentes às imagens válidas
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    print(f"Carregadas {len(df_valid)} imagens válidas. A gerar embeddings...")

    # --- Geração de Embeddings (lógica existente) ---
    extractor = DinoV3HFExtractor(model_name=args.hf_model, device=args.device)
    
    embs = []
    pbar_embed = tqdm(range(0, len(images), args.batch_size), desc="A extrair embeddings")
    for start in pbar_embed:
        end = start + args.batch_size
        batch_imgs = images[start:end]
        batch_embs = extractor.embed_pils_batch(batch_imgs)
        embs.extend(batch_embs)

    df_out = df_valid.copy()
    df_out["embedding"] = embs
    
    save_parquet(df_out, Path(args.out))
    print(f"\n[OK] Embeddings salvos em: {args.out}")
    print(f"Total de linhas processadas: {len(df_out)}")

if __name__ == "__main__":
    main()