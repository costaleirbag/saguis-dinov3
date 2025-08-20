#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np # Importar numpy
from tqdm import tqdm

from app.paths import EMBEDDINGS_DIR
# Importa a nova classe do Hugging Face
from app.vision.dinov3_extractor import DinoV3HFExtractor
from app.data.images import load_pil_from_url
from app.utils.io import save_parquet

def read_many_csvs(csv_paths: List[str], add_species: bool = True) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if add_species and "species" not in df.columns:
            sp = Path(p).stem.split("_")[0].lower()
            df["species"] = sp
        df["__srcfile__"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Nenhum CSV encontrado.")
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", default=[], help="Pode repetir --csv N vezes")
    ap.add_argument("--csv_glob", type=str, default="", help='Ex.: "data/*_location_fixed.csv"')

    # --- Argumentos modificados para Hugging Face ---
    ap.add_argument(
        "--hf_model", 
        type=str, 
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Nome do modelo DINOv3 no Hugging Face Hub."
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo a ser usado ('cuda', 'cpu', 'mps'). Auto-detecta se não for fornecido."
    )
    # --------------------------------------------------

    ap.add_argument("--out", type=str, default=str(EMBEDDINGS_DIR / "embeddings.parquet"))
    ap.add_argument("--skipped_csv", type=str, default="", help="Caminho para salvar URLs ignoradas (csv)")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dl_workers", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=0)
    args = ap.parse_args()

    csv_paths = list(args.csv)
    if args.csv_glob:
        csv_paths.extend(sorted(glob(args.csv_glob)))
    if not csv_paths:
        raise SystemExit("Informe ao menos um CSV via --csv ... ou --csv_glob ...")

    df = read_many_csvs(csv_paths, add_species=True)

    required = {"image_url", "observed_on", "latitude", "longitude", "place_state_name", "tipo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    # ---- Download paralelo (ignora erros) ----
    urls = df["image_url"].tolist()
    results = [None] * len(urls)
    skipped = []  # (idx, url, reason)

    def fetch(idx_url):
        i, u = idx_url
        img = load_pil_from_url(u)
        return i, u, img

    with ThreadPoolExecutor(max_workers=args.dl_workers) as ex:
        futures = {ex.submit(fetch, (i, u)): i for i, u in enumerate(urls)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="baixando imagens"):
            i, u, img = fut.result()
            if img is None:
                skipped.append((i, u, "download_or_decode_failed"))
            results[i] = img

    # ---- Remover linhas com falha ----
    if skipped:
        df_skipped = pd.DataFrame(skipped, columns=["index", "image_url", "reason"])
        skipped_path = Path(args.skipped_csv) if args.skipped_csv else Path(args.out).with_suffix(".skipped.csv")
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        df_skipped.to_csv(skipped_path, index=False)
        print(f"[i] Ignoradas {len(df_skipped)} imagens. Log salvo em: {skipped_path}")

        keep_mask = ~df.index.isin(df_skipped["index"])
        df = df.loc[keep_mask].reset_index(drop=True)
        images = [img for img in results if img is not None]
    else:
        images = results

    print(f"[i] Prosseguindo com {len(images)} imagens (de {len(urls)}).")

    # ---- Instancia o novo extrator do Hugging Face ----
    print(f"[i] Carregando modelo '{args.hf_model}' do Hugging Face Hub...")
    extractor = DinoV3HFExtractor(
        model_name=args.hf_model,
        device=args.device,
    )
    # ----------------------------------------------------

    embs: List[np.ndarray] = []
    partial_path = Path(args.out).with_suffix(".partial.parquet")
    pbar = tqdm(range(0, len(images), args.batch_size), desc="extraindo embeddings")
    for start in pbar:
        end = start + args.batch_size
        batch_imgs = images[start:end]
        # O método da nova classe já processa o lote inteiro
        batch_embs = extractor.embed_pils_batch(batch_imgs)
        embs.extend(batch_embs)

        if args.save_every and (start // args.batch_size + 1) % (args.save_every // args.batch_size) == 0:
            df_tmp = df.iloc[:len(embs)].copy()
            df_tmp["embedding"] = embs
            save_parquet(df_tmp, partial_path)
            pbar.set_postfix_str(f"parcial: {partial_path.name} ({len(embs)})")

    assert len(embs) == len(df), f"embeddings={len(embs)} vs df={len(df)} — tamanhos diferentes"

    df_out = df.copy()
    df_out["embedding"] = embs
    save_parquet(df_out, Path(args.out))

    try:
        if partial_path.exists():
            partial_path.unlink()
    except Exception:
        pass

    print(f"[OK] Embeddings salvos em: {args.out}")
    print(f"[i] Linhas processadas: {len(df_out)} | Ignoradas: {len(skipped)}")
    if skipped:
        print(f"[i] Veja URLs ignoradas em: {Path(args.out).with_suffix('.skipped.csv') if not args.skipped_csv else args.skipped_csv}")

if __name__ == "__main__":
    main()