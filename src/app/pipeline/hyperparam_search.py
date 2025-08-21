# src/app/pipeline/hyperparam_search.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
from pprint import pprint

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import optuna
import mlflow

from app.features.tabular import engineer_tab_features
from app.utils.io import load_parquet

# A função objective agora aceita o scale_pos_weight como argumento
def objective(trial, X, y, scale_pos_weight):
    with mlflow.start_run(nested=True):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            # --- PARÂMETRO ADICIONADO PARA LIDAR COM DESBALANCEAMENTO ---
            'scale_pos_weight': scale_pos_weight,
            # ---------------------------------------------------------
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
        mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

            model = LGBMClassifier(**params, random_state=42)
            model.fit(X_train_scaled, y_train)
            preds = model.predict_proba(X_val_scaled)[:, 1]

            if len(np.unique(y_val)) < 2:
                auc_score = 0.0
            else:
                auc_score = roc_auc_score(y_val, preds)
            scores.append(auc_score)

        final_score = float(np.mean(scores))
        mlflow.log_metric("mean_cv_auc", final_score)
        
        return final_score

def main():
    # ... (parser e carregamento de dados continuam iguais) ...
    parser = argparse.ArgumentParser(description="Hyperparameter search for Sagui classifier with MLflow tracking")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.parquet file")
    parser.add_argument("--tab_mode", type=str, default="latlon_time", help="Tabular feature engineering mode")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials to run")
    parser.add_argument("--out_dir", type=str, default="outputs/hyperparam_search", help="Directory to save results")
    parser.add_argument("--experiment_name", type=str, default="Saguis-Classifier-HPO", help="Name for the MLflow experiment")
    # Urban proximity (IBGE)
    parser.add_argument("--urban_areas_path", type=str, default=None, help="Path to IBGE urban areas dataset (e.g., .gpkg/.geojson/.shp)")
    parser.add_argument("--urban_layer", type=str, default=None, help="Layer name for GeoPackage (if applicable)")
    parser.add_argument("--urban_radius_km", type=float, default=5.0, help="Radius in km for counting nearby urban polygons")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando e preparando os dados...")
    df = load_parquet(args.embeddings)
    df['target'] = (df['tipo'] == 'H').astype(int)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df_embs = df[emb_cols]
    df_tab, tab_cols = engineer_tab_features(
        df, mode=args.tab_mode,
        urban_areas_path=args.urban_areas_path,
        urban_layer=args.urban_layer,
        urban_radius_km=args.urban_radius_km,
    )
    # Logging: confirm whether urban features were included
    urban_cols = [c for c in tab_cols if c.startswith("in_urban") or c.startswith("dist_urban_") or c.startswith("urban_count_within_")]
    
    if args.urban_areas_path:
        if urban_cols:
            print(f"[Urban] Features adicionadas: {urban_cols}")
        else:
            print("[Urban] Aviso: Nenhuma feature urbana detectada. Verifique o caminho/ camada fornecidos.")

    print("df_tab (first row, as dict):")
    pprint(df_tab.iloc[0].to_dict(), sort_dicts=False)

    X = pd.concat([df_embs, df_tab], axis=1)
    feature_cols = emb_cols + tab_cols
    X = X[feature_cols]
    y = df['target']
    
    # --- CÁLCULO DO PESO PARA DESBALANCEAMENTO ---
    counts = y.value_counts()
    scale_pos_weight = counts[0] / counts[1]
    print(f"\nClasses desbalanceadas detectadas. Usando scale_pos_weight = {scale_pos_weight:.2f}\n")
    # ---------------------------------------------
    
    mlflow.set_experiment(args.experiment_name)
    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # Comentado para usar o modo de ficheiros

    with mlflow.start_run(run_name=f"Optuna_Study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # log dataset/feature config
        mlflow.log_params({
            "tab_mode": args.tab_mode,
            "urban_areas_path": args.urban_areas_path or "",
            "urban_layer": args.urban_layer or "",
            "urban_radius_km": args.urban_radius_km,
        })
        print(f"Iniciando busca de hiperparâmetros com MLflow. Veja em http://localhost:5000")
        
        study = optuna.create_study(direction='maximize')
        # Passamos o peso calculado para a função objective
        study.optimize(
            lambda trial: objective(trial, X, y, scale_pos_weight), 
            n_trials=args.n_trials, 
            show_progress_bar=True
        )

    # ... (o resto do código para salvar os resultados continua o mesmo) ...
    print("\nBusca de hiperparâmetros concluída.")
    print(f"Melhor trial (AUC): {study.best_value:.4f}")
    print("Melhores parâmetros:")
    print(json.dumps(study.best_params, indent=2))

    results_path = out_dir / "study_results.joblib"
    joblib.dump(study, results_path)
    print(f"\nResultados completos do estudo salvos em: {results_path}")

    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Melhores parâmetros salvos em: {best_params_path}")
    
    features_path = out_dir / "feature_list.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Lista de features salva em: {features_path}")

    # Save tabular/urban config for reproducibility
    tabcfg_path = out_dir / "tabular_config.json"
    with open(tabcfg_path, 'w') as f:
        json.dump({
            "tab_mode": args.tab_mode,
            "urban_areas_path": args.urban_areas_path,
            "urban_layer": args.urban_layer,
            "urban_radius_km": args.urban_radius_km,
        }, f, indent=2)
    print(f"Config de features tabulares salva em: {tabcfg_path}")


if __name__ == "__main__":
    main()