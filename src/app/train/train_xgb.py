#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
from app.paths import MODELS_DIR
from app.data.tabular import build_tabular_matrix
from app.utils.io import load_parquet

def stack_features(df: pd.DataFrame, fit_encoders: bool, enc_dir: Path):
    y = (df["tipo"] == "H").astype(int).values
    X_tab, _ = build_tabular_matrix(df, fit=fit_encoders, enc_dir=enc_dir)
    X_img = np.vstack(df["embedding"].values)
    X = np.hstack([X_img, X_tab])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True, help="Parquet com coluna 'embedding'")
    ap.add_argument("--enc_dir", type=str, default=str(MODELS_DIR / "encoders"))
    ap.add_argument("--model_out", type=str, default=str(MODELS_DIR / "xgb_model.json"))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = load_parquet(Path(args.embeddings))
    X, y = stack_features(df, fit_encoders=True, enc_dir=Path(args.enc_dir))
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.test_size,
                                          stratify=y, random_state=args.random_state)

    clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=0,
    )
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xva)[:, 1]
    yhat = (p >= 0.5).astype(int)
    auc = roc_auc_score(yva, p)
    acc = accuracy_score(yva, yhat)

    print(f"AUC: {auc:.4f} | ACC: {acc:.4f}")
    print(classification_report(yva, yhat, target_names=["N-H", "H"]))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(args.model_out)
    print(f"[OK] Modelo salvo em: {args.model_out}")

if __name__ == "__main__":
    main()
