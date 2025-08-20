import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from typing import Tuple
from app.utils.io import dump, load

@dataclass
class TabularEncoders:
    scaler: StandardScaler
    ohe: OneHotEncoder

    def save(self, dirpath):
        dump(self.scaler, dirpath / "scaler.joblib")
        dump(self.ohe, dirpath / "ohe.joblib")

    @staticmethod
    def load(dirpath):
        return TabularEncoders(
            scaler=load(dirpath / "scaler.joblib"),
            ohe=load(dirpath / "ohe.joblib"),
        )

def build_tabular_matrix(df: pd.DataFrame, fit: bool, enc_dir) -> Tuple[np.ndarray, TabularEncoders]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["observed_on"], dayfirst=True, errors="coerce").dt.year.fillna(-1).astype(int)

    num_cols = ["latitude", "longitude", "year"]
    cat_cols = ["place_state_name"]

    if fit:
        scaler = StandardScaler().fit(df[num_cols])
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[cat_cols])
        encs = TabularEncoders(scaler, ohe)
        encs.save(enc_dir)
    else:
        encs = TabularEncoders.load(enc_dir)

    num = encs.scaler.transform(df[num_cols])
    cat = encs.ohe.transform(df[cat_cols])
    X_tab = np.hstack([num, cat])
    return X_tab, encs
