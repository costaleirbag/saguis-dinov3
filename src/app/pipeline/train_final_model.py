# src/app/pipeline/train_final_model.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

from app.features.tabular import engineer_tab_features
from app.utils.io import load_parquet

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the final classifier model")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.parquet file")
    parser.add_argument("--hpo_dir", type=str, required=True, help="Directory containing hyperparameter search results (best_params.json, feature_list.json)")
    parser.add_argument("--model_out_dir", type=str, default="outputs/final_model", help="Directory to save the final trained model and artifacts")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    args = parser.parse_args()

    hpo_dir = Path(args.hpo_dir)
    model_out_dir = Path(args.model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Carregar Artefatos da Otimização ---
    print("Carregando os melhores parâmetros e a lista de features da otimização...")
    with open(hpo_dir / "best_params.json", 'r') as f:
        best_params = json.load(f)
    with open(hpo_dir / "feature_list.json", 'r') as f:
        feature_cols = json.load(f)

    # --- 2. Preparar os Dados ---
    print("Carregando e preparando o dataset completo...")
    df = load_parquet(args.embeddings)
    df['target'] = (df['tipo'] == 'H').astype(int)

    # Recriar as features exatamente como na otimização
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df_embs = df[emb_cols]
    df_tab, _ = engineer_tab_features(df, mode="latlon_time")
    X = pd.concat([df_embs, df_tab], axis=1)[feature_cols] # Garante a ordem correta
    y = df['target']

    # --- 3. Divisão Final de Treino/Teste ---
    print(f"Dividindo os dados: {1-args.test_size:.0%} para treino, {args.test_size:.0%} para teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # --- 4. Treinar o Modelo Final ---
    print("Treinando o modelo final com os melhores hiperparâmetros no conjunto de treino completo...")
    
    # Calcular o peso para desbalanceamento no conjunto de treino
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]
    
    # Criar o pipeline final: Scaler + Classificador
    scaler = StandardScaler()
    model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)

    # --- 5. Avaliar no Conjunto de Teste (Hold-out) ---
    print("\n--- Avaliação Final no Conjunto de Teste ---")
    preds_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Encontrar o melhor limiar (threshold) no conjunto de treino para maximizar o F1-score
    train_preds_proba = pipeline.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_train, train_preds_proba > t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    preds_binary = (preds_proba >= best_threshold).astype(int)

    # Métricas finais
    final_auc = roc_auc_score(y_test, preds_proba)
    final_accuracy = accuracy_score(y_test, preds_binary)
    
    print(f"Melhor Limiar (Threshold) encontrado: {best_threshold:.4f}")
    print(f"AUC Final: {final_auc:.4f}")
    print(f"Acurácia Final: {final_accuracy:.4f}\n")
    
    print("Relatório de Classificação:")
    print(classification_report(y_test, preds_binary, target_names=["N-H (0)", "H (1)"]))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, preds_binary))

    # --- 6. Salvar o Modelo Final ---
    model_path = model_out_dir / "final_model.joblib"
    
    artifacts = {
        "pipeline": pipeline,
        "feature_list": feature_cols,
        "best_threshold": best_threshold,
        "metrics": {
            "auc": final_auc,
            "accuracy": final_accuracy
        }
    }
    joblib.dump(artifacts, model_path)
    print(f"\n[OK] Modelo final e artefactos salvos em: {model_path}")

if __name__ == "__main__":
    main()