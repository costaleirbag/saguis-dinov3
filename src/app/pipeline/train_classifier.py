# src/app/pipeline/train_classifier.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from xgboost import XGBClassifier

# permitir "python -m app.pipeline.train_classifier"
sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.features.tabular import engineer_tab_features  # noqa

# ------------ Args ------------
def parse_args():
    ap = argparse.ArgumentParser(description="Treina classificador com embeddings DINOv3 + tabular.")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", dest="out_dir", default="outputs/models/run")
    ap.add_argument("--label_col", default="tipo")
    ap.add_argument("--model", choices=["xgb","logreg"], default="xgb")

    # PCA e padronização
    ap.add_argument("--pca_components", type=int, default=128, help="PCA só nos embeddings (0=desliga)")
    ap.add_argument("--scale_all", action="store_true", help="padroniza X final (para logreg geralmente sim; XGB não precisa)")

    # CV e seeds
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0)

    # Tabular
    ap.add_argument("--tab_mode", choices=["latlon_time","full","none"], default="latlon_time")

    # Limiar
    ap.add_argument("--threshold_opt", choices=["none","f1","youden","balanced_accuracy"], default="none")

    # XGB params (single run)
    ap.add_argument("--xgb_params", type=str, default="",
                    help='JSON com hiperparâmetros do XGBoost (ex.: \'{"n_estimators":900,"max_depth":4,...}\')')

    # GRID SEARCH
    ap.add_argument("--pca_grid", type=str, default="", help="lista de ints (ex.: '0,64,128,256')")
    ap.add_argument("--xgb_grid", type=str, default="", help="JSON array de dicionários (ex.: '[{...},{...}]')")
    ap.add_argument("--grid_metric", choices=["auc","f1"], default="auc",
                    help="métrica alvo p/ escolher melhor configuração no grid")

    return ap.parse_args()

# ------------ utilidades ------------
def best_threshold(y_true, y_prob, mode: str) -> float:
    if mode == "none":
        return 0.5
    thr_grid = np.linspace(0.05, 0.95, 181)
    best = 0.5
    best_s = -1.0
    for t in thr_grid:
        y_pred = (y_prob >= t).astype(int)
        if mode == "f1":
            s = f1_score(y_true, y_pred, zero_division=0)
        elif mode in ("youden","balanced_accuracy"):
            s = balanced_accuracy_score(y_true, y_pred)
        else:
            s = 0.0
        if s > best_s:
            best_s, best = s, float(t)
    return best

def make_xgb(params: Dict[str, Any]) -> XGBClassifier:
    defaults = dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=2.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=os.cpu_count() or 8,
        eval_metric="logloss",
    )
    defaults.update(params or {})
    return XGBClassifier(**defaults)

def make_logreg(class_weight=None) -> LogisticRegression:
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        n_jobs=1,
        solver="lbfgs",
    )

def split_features(df: pd.DataFrame, pca: PCA|None, scale_all: bool,
                   tab_mode: str, args_random_state: int):
    # alvo binário (H positivo)
    y = (df["tipo"].astype(str) == "H").astype(int).to_numpy()

    # embeddings: detecta colunas começando com "emb_"
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        # coluna única "embedding" -> expandir
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in df["embedding"].to_numpy()])
        emb_cols = [f"emb_{i}" for i in range(arr.shape[1])]
        E = arr
    else:
        E = df[emb_cols].to_numpy().astype(np.float32)

    # tabular
    T_df, tab_cols = engineer_tab_features(df, mode=tab_mode)
    T = T_df.to_numpy().astype(np.float32)

    # concat depois (PCA só nos embeddings)
    return y, E, emb_cols, T, tab_cols

def apply_pca(E: np.ndarray, n_comp: int, random_state: int):
    if n_comp and 0 < n_comp < E.shape[1]:
        pca = PCA(n_components=n_comp, random_state=random_state)
        E2 = pca.fit_transform(E)
        info = {"n_components": int(n_comp), "explained_var_ratio_sum": float(pca.explained_variance_ratio_.sum())}
        return E2, pca, info
    return E, None, None

def do_cv(X: np.ndarray, y: np.ndarray, model_builder, cv_folds: int, seed: int, thr_mode: str):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    per_fold = []
    for k, (tr, va) in enumerate(skf.split(X, y), start=1):
        clf = model_builder()
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[va])[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X[va])
        thr = best_threshold(y[va], prob, thr_mode)
        pred = (prob >= thr).astype(int)
        acc = accuracy_score(y[va], pred)
        f1  = f1_score(y[va], pred, zero_division=0)
        auc = roc_auc_score(y[va], prob)
        per_fold.append({"fold":k, "acc":acc, "f1":f1, "auc":auc, "thr":thr})
        oof[va] = prob
        print(f"[fold {k}] acc={acc:.4f} | f1={f1:.4f} | auc={auc:.4f} | thr={thr:.3f}")
    # métrica global com um único thr
    thr_g = best_threshold(y, oof, thr_mode)
    yhat = (oof >= thr_g).astype(int)
    return {
        "oof_prob": oof,
        "folds": per_fold,
        "oof_acc": accuracy_score(y, yhat),
        "oof_f1": f1_score(y, yhat, zero_division=0),
        "oof_auc": roc_auc_score(y, oof),
        "thr_global": thr_g,
    }

def save_feature_log(out_dir: Path, used_cols: List[str], emb_cols: List[str], tab_cols: List[str]):
    info = {
        "n_features_total": len(used_cols),
        "n_emb_features": len([c for c in used_cols if c.startswith("emb_") or c.startswith("pca_")]),
        "n_tab_features": len([c for c in used_cols if c not in emb_cols and not c.startswith("pca_")]),
        "embedding_cols_sample": used_cols[: min(10, len(used_cols))],
        "tab_cols": tab_cols,
        "all_feature_names": used_cols,
    }
    (out_dir / "features_used.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

# ------------ Main ------------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Lendo embeddings: {args.embeddings}")
    df = pd.read_parquet(args.embeddings)
    if args.max_rows and args.max_rows > 0:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.random_state)

    # split base
    y, E_base, emb_cols_base, T_base, tab_cols_base = split_features(df, None, args.scale_all, args.tab_mode, args.random_state)

    # GRID?
    has_grid = bool(args.pca_grid or args.xgb_grid)
    if has_grid:
        # parse grids
        pca_grid = [int(x) for x in args.pca_grid.split(",")] if args.pca_grid else [args.pca_components]
        xgb_grid = json.loads(args.xgb_grid) if args.xgb_grid else [json.loads(args.xgb_params) if args.xgb_params else {}]

        results = []
        best = None

        for n_comp in pca_grid:
            E, pca, pca_info = apply_pca(E_base, n_comp, args.random_state)
            X = np.hstack([E, T_base]).astype(np.float32)

            scaler = None
            if args.scale_all:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            used_cols = [f"{'pca' if pca else 'emb'}_{i}" for i in range(E.shape[1])] + tab_cols_base
            save_feature_log(out_dir, used_cols, emb_cols_base, tab_cols_base)

            for params in xgb_grid:
                def builder():
                    if args.model == "logreg":
                        return make_logreg(class_weight=None)
                    return make_xgb(params)

                print(f"\n[i] GRID — PCA={n_comp} | params={params}")
                res = do_cv(X, y, builder, args.cv_folds, args.random_state, args.threshold_opt)
                row = {
                    "pca": n_comp,
                    "params": json.dumps(params),
                    "oof_acc": res["oof_acc"],
                    "oof_f1": res["oof_f1"],
                    "oof_auc": res["oof_auc"],
                    "thr": res["thr_global"],
                }
                results.append(row)

                # escolher melhor
                score = row["oof_auc"] if args.grid_metric == "auc" else row["oof_f1"]
                if best is None or score > (best["oof_auc"] if args.grid_metric=="auc" else best["oof_f1"]):
                    best = row | {"scaler": scaler, "pca_info": pca_info, "pca_obj": pca, "used_cols": used_cols, "params_dict": params}

        # salvar CSV com ranking
        df_res = pd.DataFrame(results).sort_values(f"oof_{args.grid_metric}", ascending=False)
        df_res.to_csv(out_dir / "grid_results.csv", index=False)
        print("\n[GRID] Top 5:\n", df_res.head(5).to_string(index=False))

        # treinar final com melhor config
        best_pca = best["pca"]
        best_params = best["params_dict"]
        print(f"\n[GRID] Melhor config: PCA={best_pca} | metric={args.grid_metric} -> {best['oof_'+args.grid_metric]:.4f}")
        # remonta X com melhor PCA
        E_best, pca_obj, _ = apply_pca(E_base, best_pca, args.random_state)
        X_best = np.hstack([E_best, T_base]).astype(np.float32)
        scaler = None
        if args.scale_all:
            scaler = StandardScaler()
            X_best = scaler.fit_transform(X_best)

        final = make_xgb(best_params) if args.model == "xgb" else make_logreg()
        final.fit(X_best, y)

        dump({
            "model": final,
            "scaler": scaler,
            "pca": pca_obj,
            "used_cols": best["used_cols"],
            "tab_mode": args.tab_mode,
            "threshold": best["thr"],
            "best_params": best_params,
            "grid_metric": args.grid_metric,
        }, out_dir / "best_model.joblib")

        # meta
        (out_dir / "meta.json").write_text(json.dumps({
            "grid_metric": args.grid_metric,
            "best": {k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v) for k,v in best.items()
                     if k in ("pca","oof_acc","oof_f1","oof_auc","thr","params")},
        }, indent=2))
        print(f"[OK] Best model salvo em: {out_dir/'best_model.joblib'}")
        print(f"[OK] Grid results em: {out_dir/'grid_results.csv'}")
        return

    # ------ Single run (sem grid) ------
    # PCA
    E, pca, pca_info = apply_pca(E_base, args.pca_components, args.random_state)
    X = np.hstack([E, T_base]).astype(np.float32)

    scaler = None
    if args.scale_all:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    used_cols = [f"{'pca' if pca else 'emb'}_{i}" for i in range(E.shape[1])] + tab_cols_base
    save_feature_log(out_dir, used_cols, emb_cols_base, tab_cols_base)

    # modelo
    if args.model == "logreg":
        model_builder = lambda: make_logreg(class_weight=None)
    else:
        params = json.loads(args.xgb_params) if args.xgb_params else {}
        model_builder = lambda: make_xgb(params)

    print(f"[i] Treinando com {args.cv_folds}-fold CV (n={len(y)} amostras).")
    res = do_cv(X, y, model_builder, args.cv_folds, args.random_state, args.threshold_opt)
    print(f"\n[OOF] acc={res['oof_acc']:.4f} | f1={res['oof_f1']:.4f} | auc={res['oof_auc']:.4f} | thr_global={res['thr_global']:.3f}")

    final = model_builder(); final.fit(X, y)
    dump({
        "model": final,
        "scaler": scaler,
        "pca": pca,
        "used_cols": used_cols,
        "tab_mode": args.tab_mode,
        "threshold": res["thr_global"],
        "params": json.loads(args.xgb_params) if args.xgb_params else {},
    }, out_dir / "model.joblib")

    meta = {
        "embeddings_file": args.embeddings,
        "label_col": args.label_col,
        "model_type": args.model,
        "pca": pca_info,
        "scale_all": args.scale_all,
        "cv_folds": args.cv_folds,
        "tab_mode": args.tab_mode,
        "threshold_opt": args.threshold_opt,
        "oof": {"acc": res["oof_acc"], "f1": res["oof_f1"], "auc": res["oof_auc"], "thr": res["thr_global"]},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[OK] Modelo salvo em: {out_dir/'model.joblib'}")
    print(f"[OK] Metadados em: {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
