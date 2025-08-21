# src/app/features/urban_proximity.py
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "geo"
CACHE_PATH = DATA_DIR / "urban_centers.gpkg"

# Baixa malha urbana do OSM via GeoFabrik (Sudeste do Brasil)
def fetch_urban_centers():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        return gpd.read_file(CACHE_PATH)

    # usar OSMnx para baixar localidades
    import osmnx as ox
    print("Baixando localidades urbanas do OSM...")
    tags = {"place": ["city", "town", "village", "suburb"]}
    gdf = ox.geometries_from_place("Sudeste, Brazil", tags=tags)

    # pegar apenas centroides
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf.centroid
    gdf = gdf[["geometry", "place"]]
    gdf.to_file(CACHE_PATH, driver="GPKG")
    return gdf

def add_urban_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entra: df com colunas latitude, longitude
    Sai: df com nova coluna 'dist_nearest_city_km'
    """
    centers = fetch_urban_centers()

    results = []
    for _, row in df.iterrows():
        pt = (row["latitude"], row["longitude"])
        min_dist = centers["geometry"].apply(
            lambda g: geodesic(pt, (g.y, g.x)).km
        ).min()
        results.append(min_dist)

    df["dist_nearest_city_km"] = results
    return df
