# src/app/features/tabular.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime

def _parse_date_series(s: pd.Series) -> pd.Series:
    def _try_parse(x):
        try:
            return datetime.strptime(str(x), "%d/%m/%Y")
        except Exception:
            return pd.NaT
    return s.astype(str).apply(_try_parse)

def engineer_tab_features(
    df: pd.DataFrame,
    mode: str = "latlon_time",  # "latlon_time" | "full" | "none"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Gera features tabulares.
    - latlon_time: usa somente latitude, longitude e derivados de observed_on
    - full: inclui também one-hot de estado (compatível com versão anterior)
    - none: retorna DataFrame vazio
    """
    if mode == "none":
        return pd.DataFrame(index=df.index), []

    # bases
    out = pd.DataFrame(index=df.index)
    if "latitude" in df.columns:
        out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    else:
        out["latitude"] = 0.0
    if "longitude" in df.columns:
        out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    else:
        out["longitude"] = 0.0

    # tempo
    if "observed_on" in df.columns:
        dts = _parse_date_series(df["observed_on"])
    else:
        dts = pd.Series(pd.NaT, index=df.index)

    out["year"] = dts.dt.year
    out["month"] = dts.dt.month
    out["dayofyear"] = dts.dt.dayofyear

    two_pi = 2 * np.pi
    out["month_sin"] = np.sin(two_pi * (out["month"].fillna(0) / 12))
    out["month_cos"] = np.cos(two_pi * (out["month"].fillna(0) / 12))
    out["doy_sin"]   = np.sin(two_pi * (out["dayofyear"].fillna(0) / 366))
    out["doy_cos"]   = np.cos(two_pi * (out["dayofyear"].fillna(0) / 366))

    cols = [
        "latitude","longitude","year","month","dayofyear",
        "month_sin","month_cos","doy_sin","doy_cos",
    ]

    if mode == "full":
        states = pd.get_dummies(
            df.get("place_state_name", pd.Series(dtype=str)).astype(str),
            prefix="state"
        )
        out = pd.concat([out, states], axis=1)
        cols = cols + [c for c in out.columns if c.startswith("state_")]

    # numérico + NaN->0
    out = out.fillna(0.0).astype(float)
    return out[cols], cols
