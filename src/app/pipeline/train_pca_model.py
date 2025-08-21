# src/app/pipeline/train_pca_model.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

from app.features.tabular import engineer_tab_features
from app.utils.io import load_parquet

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the final classifier model with PCA")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.parquet file")
    parser.add_argument("--hpo_dir", type=str, required=True, help="Directory containing hyperparameter search results")
    parser.add_argument("--model_out_dir", type=str, default="outputs/final_model_pca", help="Directory to save the final PCA model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset for the test split")
    parser.add_argument("--pca_components", type=int, default=128, help="Number of principal components to keep")
    args = parser.parse_args()

    hpo_dir = Path(args.hpo_dir)
    model_out_dir = Path(args.model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando artefactos da otimização...")
    with open(hpo_dir / "best_params.json", 'r') as f:
        best_params = json.load(f)
    
    print("Carregando e preparando o dataset...")
    df = load_parquet(args.embeddings)
    df['target'] = (df['tipo'] == 'H').astype(int)

    # --- LÓGICA DE PREPARAÇÃO DE DADOS CORRIGIDA ---
    # 1. Converte a coluna 'embedding' numa matriz 2D
    embedding_matrix = np.vstack(df["embedding"].values)
    
    # 2. Cria um DataFrame a partir da matriz com nomes de colunas padronizados
    emb_cols = [f"emb_{i}" for i in range(embedding_matrix.shape[1])]
    df_embs = pd.DataFrame(embedding_matrix, columns=emb_cols, index=df.index)
    # ---------------------------------------------------

    df_tab, tab_cols = engineer_tab_features(df, mode="latlon_time")
    
    print("Dividindo os dados em conjuntos de treino e teste...")
    X_train_embs, X_test_embs, X_train_tab, X_test_tab, y_train, y_test = train_test_split(
        df_embs, df_tab, df['target'], test_size=args.test_size, stratify=df['target'], random_state=42
    )

    print(f"Construindo pipeline com PCA para {args.pca_components} componentes...")
    if args.pca_components == -1:
        # No PCA, just scaling
        embedding_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        emb_feature_names = emb_cols
    else:
        embedding_pipeline = Pipeline([
            ('pca', PCA(n_components=args.pca_components, random_state=42)),
            ('scaler', StandardScaler())
        ])
        emb_feature_names = [f"pca_{i}" for i in range(args.pca_components)]
    tabular_pipeline = Pipeline([('scaler', StandardScaler())])

    print("Treinando os pipelines de pré-processamento...")
    X_train_embs_processed = embedding_pipeline.fit_transform(X_train_embs)
    X_train_tab_processed = tabular_pipeline.fit_transform(X_train_tab)
    
    X_train_final = np.hstack([X_train_embs_processed, X_train_tab_processed])

    print("Treinando o modelo final com os melhores hiperparâmetros...")
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]
    
    model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)
    model.fit(X_train_final, y_train)

    print("\n--- Avaliação Final no Conjunto de Teste ---")
    
    X_test_embs_processed = embedding_pipeline.transform(X_test_embs)
    X_test_tab_processed = tabular_pipeline.transform(X_test_tab)
    X_test_final = np.hstack([X_test_embs_processed, X_test_tab_processed])
    
    preds_proba = model.predict_proba(X_test_final)[:, 1]
    
    train_preds_proba = model.predict_proba(X_train_final)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_train, train_preds_proba > t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    preds_binary = (preds_proba >= best_threshold).astype(int)

    final_auc = roc_auc_score(y_test, preds_proba)
    print(f"Melhor Limiar (Threshold) encontrado: {best_threshold:.4f}")
    print(f"AUC Final com PCA: {final_auc:.4f}\n")
    
    print("Relatório de Classificação:")
    print(classification_report(y_test, preds_binary, target_names=["N-H (0)", "H (1)"]))

    model_path = model_out_dir / "final_model.joblib"
    
    artifacts = {
        "embedding_pipeline": embedding_pipeline,
        "tabular_pipeline": tabular_pipeline,
        "model": model,
        "feature_names": {
            "embeddings_pca": emb_feature_names,
            "tabular": tab_cols
        },
        "best_threshold": best_threshold,
        "metrics": {"auc": final_auc}
    }
    joblib.dump(artifacts, model_path)
    print(f"\n[OK] Modelo final com PCA e artefactos salvos em: {model_path}")

if __name__ == "__main__":
    main()