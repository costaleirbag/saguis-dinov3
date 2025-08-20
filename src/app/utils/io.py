from pathlib import Path
import joblib
import pandas as pd

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load(path: Path):
    return joblib.load(path)
