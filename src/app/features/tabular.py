# src/app/features/tabular.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime

# Optional geospatial stack (GeoPandas/Shapely). Kept optional to avoid hard dependency on import.
try:  # lightweight guard; raise helpful error at callsites if missing
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:  # pragma: no cover - optional dep
    gpd = None  # type: ignore
    Point = None  # type: ignore

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parses a Series of strings into datetime objects.

    Args:
        s: A pandas Series with date-like strings.

    Returns:
        A pandas Series with datetime objects, where unparseable
        dates are converted to NaT.
    """
    return pd.to_datetime(s, format="%d/%m/%Y", errors='coerce')

def engineer_tab_features(
    df: pd.DataFrame,
    mode: str = "latlon_time",  # "latlon_time" | "full" | "none"
    *,
    # Urban proximity (IBGE polygons) – pass either a preloaded GeoDataFrame or a path to load
    urban_gdf: object | None = None,
    urban_areas_path: str | None = None,
    urban_layer: str | None = None,
    urban_radius_km: float = 5.0,
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
        dts = pd.to_datetime(_parse_date_series(df["observed_on"]), errors="coerce")
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

    # Urban proximity features (IBGE polygons)
    if urban_gdf is None and urban_areas_path:
        urban_gdf = load_ibge_urban_areas(urban_areas_path, layer=urban_layer)
    if urban_gdf is not None:
        urb_df, urb_cols = engineer_urban_proximity_features(df, urban_gdf, radius_km=urban_radius_km)
        out = pd.concat([out, urb_df], axis=1)
        cols = cols + urb_cols

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


# =============================
# Urban proximity: IBGE support
# =============================
def load_ibge_urban_areas(path: str, *, layer: str | None = None, crs_out: str = "EPSG:4326"):
    """
    Load IBGE urban areas polygons (e.g., "Áreas Urbanizadas" GeoPackage/GeoJSON/Shapefile).
    - path: local file path to a GeoPackage (.gpkg), Shapefile (.shp), or GeoJSON with polygons.
    - layer: optional layer name (for .gpkg with multiple layers).
    - returns a GeoDataFrame with columns [urban_name, urban_pop?, geometry] in EPSG:4326.

    Note: This function does not download data. Place the IBGE dataset under data/geo/ and pass the path.
    """
    if gpd is None:  # pragma: no cover - optional dep
        raise ImportError("Geospatial dependencies missing. Install: poetry add geopandas shapely pyproj")

    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(crs_out).copy()

    # Try to normalize name/population if present; otherwise leave as NaN/None
    name_col = next((c for c in [
        "nome","name","NM_MUNICIP","NM_MUN","NM_UF","NOME","NOME_MUN",
        "NM_LOCALID","NM_SUBDISTR","NM_BAIRRO","NM_AREA_URB","NM_ARURB"
    ] if c in gdf.columns), None)
    pop_col = next((c for c in [
        "POPULACAO","POP","POP_TOT","POP2010","POP_2010","pop","population"
    ] if c in gdf.columns), None)

    gdf["urban_name"] = gdf[name_col].astype(str) if name_col else None
    gdf["urban_pop"] = pd.to_numeric(gdf[pop_col], errors="coerce") if pop_col else np.nan
    # Keep only needed columns
    keep = [c for c in ["urban_name", "urban_pop", "geometry"] if c in gdf.columns] + ["geometry"]
    keep = list(dict.fromkeys(keep))  # unique, preserve order
    return gdf[keep]


def engineer_urban_proximity_features(
    df: pd.DataFrame,
    urban_gdf,  # GeoDataFrame in EPSG:4326 with polygons
    *,
    crs_metric: str = "EPSG:3857",
    radius_km: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute urban proximity features using IBGE urban areas polygons.

    Features created:
    - in_urban: 1.0 if the point lies inside any urban polygon; else 0.0
    - dist_urban_poly_m: distance in meters to nearest urban polygon (0 if inside)
    - dist_urban_centroid_m: distance in meters to nearest urban polygon centroid
    - urban_count_within_{R}km: number of polygons intersecting a radius R km buffer
    """
    if gpd is None or Point is None:  # pragma: no cover - optional dep
        raise ImportError("Geospatial dependencies missing. Install: poetry add geopandas shapely pyproj")
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("df must contain 'latitude' and 'longitude' columns")

    # Points GeoDataFrame in WGS84
    pts = gpd.GeoDataFrame(
        df[["latitude", "longitude"]].copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(df["latitude"], df["longitude"])],
        crs="EPSG:4326",
        index=df.index,
    )

    # Ensure urban_gdf in WGS84 and also make metric versions
    urban_wgs = urban_gdf.to_crs("EPSG:4326") if getattr(urban_gdf, "crs", None) != "EPSG:4326" else urban_gdf
    pts_m = pts.to_crs(crs_metric)
    urban_m = urban_wgs.to_crs(crs_metric)

    # Inside urban polygon
    within = gpd.sjoin(pts, urban_wgs[["geometry"]], how="left", predicate="within")
    in_urban = within.index_right.notna().astype(float).reindex(df.index).fillna(0.0)

    # Distance to nearest polygon (meters) using sjoin_nearest on metric CRS
    nearest_poly = gpd.sjoin_nearest(pts_m, urban_m[["geometry"]], how="left", distance_col="_dist_poly_m")
    dist_poly_m = nearest_poly["_dist_poly_m"].astype(float).reindex(df.index)
    # Zero-out distances for points that are inside
    dist_poly_m = dist_poly_m.where(in_urban == 0.0, other=0.0)

    # Distance to nearest centroid (meters)
    urban_centroids_m = urban_m.copy()
    # centroid on projected CRS
    urban_centroids_m["geometry"] = urban_centroids_m.geometry.centroid
    nearest_ctr = gpd.sjoin_nearest(pts_m, urban_centroids_m[["geometry"]], how="left", distance_col="_dist_ctr_m")
    dist_centroid_m = nearest_ctr["_dist_ctr_m"].astype(float).reindex(df.index)

    # Count urban polygons intersecting a buffer of R km
    buffer_m = pts_m.buffer(radius_km * 1000.0)
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer_m, crs=crs_metric, index=df.index)
    buf_join = gpd.sjoin(buffer_gdf, urban_m[["geometry"]], how="left", predicate="intersects")
    counts = buf_join.groupby(level=0).size().reindex(df.index).fillna(0).astype(int)

    out = pd.DataFrame(index=df.index)
    out["in_urban"] = in_urban
    out["dist_urban_poly_m"] = (
        dist_poly_m.replace([np.inf, -np.inf], np.nan).fillna(1e7).astype(float)
    )
    out["dist_urban_centroid_m"] = (
        dist_centroid_m.replace([np.inf, -np.inf], np.nan).fillna(1e7).astype(float)
    )
    count_col = f"urban_count_within_{int(radius_km)}km"
    out[count_col] = counts.astype(float)

    cols = ["in_urban", "dist_urban_poly_m", "dist_urban_centroid_m", count_col]
    return out[cols], cols
