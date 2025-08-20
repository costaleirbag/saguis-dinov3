# src/app/cli/predict_one.py
import argparse
import json
from app.inference.predictor import SaguiPredictor, PredictorConfig

def parse_args():
    ap = argparse.ArgumentParser("Inferência única (URL + data + lat/lon)")
    ap.add_argument("--model", required=True, help="Caminho para best_model.joblib")
    # --- Argumento atualizado ---
    ap.add_argument(
        "--hf_model",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Nome do modelo DINOv3 no Hugging Face Hub"
    )
    # -----------------------------
    ap.add_argument("--img_url", required=True)
    ap.add_argument("--date", required=True, help="ex.: 23/10/2009 ou 2009-10-23")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    # --- Instanciação do Config corrigida ---
    cfg = PredictorConfig(
        model_path=args.model,
        hf_model=args.hf_model,
    )
    # ----------------------------------------
    pred = SaguiPredictor(cfg)
    out = pred.predict(
        image=args.img_url,
        observed_on=args.date,
        latitude=args.lat,
        longitude=args.lon,
        return_intermediate=False,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()