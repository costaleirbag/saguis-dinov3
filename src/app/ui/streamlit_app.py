# src/app/ui/streamlit_app.py
import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import streamlit as st
from streamlit_folium import st_folium
import folium
from PIL import Image
import requests

from app.inference.predictor import SaguiPredictor, PredictorConfig
from app.data.images import load_pil_from_url

# --------- Config ---------
SE_BOUNDS = {
    "min_lat": -25.5, "max_lat": -14.0,
    "min_lon": -53.0, "max_lon": -39.0
}
SE_CENTER = (-21.0, -44.0)
SE_ZOOM = 5

# --- Função de carregamento do modelo ATUALIZADA ---
@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str, hf_model: str, device_prefer: str):
    """Carrega e armazena em cache a instância do predictor."""
    cfg = PredictorConfig(
        model_path=model_path,
        hf_model=hf_model,
        device_prefer=device_prefer,
    )
    return SaguiPredictor(cfg)
# ----------------------------------------------------

def clamp_to_bounds(lat: float, lon: float) -> Tuple[float, float]:
    """Garante que as coordenadas estejam dentro dos limites definidos."""
    lat = max(min(lat, SE_BOUNDS["max_lat"]), SE_BOUNDS["min_lat"])
    lon = max(min(lon, SE_BOUNDS["max_lon"]), SE_BOUNDS["min_lon"])
    return lat, lon

def se_contains(lat: float, lon: float) -> bool:
    """Verifica se as coordenadas estão na região Sudeste."""
    return (SE_BOUNDS["min_lat"] <= lat <= SE_BOUNDS["max_lat"]) and (SE_BOUNDS["min_lon"] <= lon <= SE_BOUNDS["max_lon"])

def nice_location_label(lat: float, lon: float) -> str:
    """Cria um rótulo formatado para as coordenadas."""
    in_se = se_contains(lat, lon)
    return f"Lat {lat:.5f}, Lon {lon:.5f}" + (" • Sudeste do Brasil" if in_se else "")

@st.cache_data(show_spinner="Buscando localização...")
def geocode_city(query: str) -> Optional[Tuple[float, float]]:
    """Converte nome de cidade em (lat, lon) via Nominatim API."""
    try:
        headers = {"User-Agent": "Sagui-DINOv3-Streamlit-Demo/1.0 (seu-email@exemplo.com)"}
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com o serviço de geolocalização: {e}")
        return None
    except (KeyError, IndexError):
        return None

# --------- UI ---------
st.set_page_config(page_title="Saguis DINOv3 - Demo", layout="wide")
st.title("Saguis (H vs N-H) • DINOv3 + XGBoost")

# --- Barra Lateral ATUALIZADA ---
with st.sidebar:
    st.header("Configuração do modelo")
    model_path = st.text_input("Caminho do best_model.joblib", "outputs/models/grid_overnight_20250820_0225/best_model.joblib")
    
    # Opções de modelo DINOv3 do Hugging Face
    hf_model_options = [
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ]
    hf_model = st.selectbox("Modelo DINOv3 (Hugging Face)", hf_model_options, index=0)

    device_prefer = st.selectbox("Dispositivo", ["mps", "cuda", "cpu"], help="Selecione 'cuda' para placas Nvidia, 'mps' para Mac M1/M2/M3, ou 'cpu'.")
    st.caption("A detecção de dispositivo é automática, mas você pode forçar uma opção aqui.")
# ------------------------------------

# --- Chamada ATUALIZADA para carregar o predictor ---
try:
    predictor = load_predictor(model_path, hf_model, device_prefer)
except Exception as e:
    st.error(f"Falha ao carregar o modelo: {e}")
    st.stop()
# ---------------------------------------------------

col_left, col_right = st.columns([0.55, 0.45], gap="large")

with col_left:
    st.subheader("Entrada")
    image_url = st.text_input("URL da imagem", "https://inaturalist-open-data.s3.amazonaws.com/photos/59806858/medium.png")
    date_str = st.text_input("Data do registro", "23/10/2009", help="Aceita DD/MM/AAAA ou AAAA-MM-DD")

    input_mode = st.radio(
        "Como informar a localização?",
        ["Clicar no mapa", "Digitar coordenadas", "Buscar por cidade"],
        horizontal=True
    )

    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lon = -22.90, -43.20

    if input_mode == "Clicar no mapa":
        fmap = folium.Map(location=SE_CENTER, zoom_start=SE_ZOOM, control_scale=True, tiles="OpenStreetMap")
        folium.Marker(
            location=(st.session_state.lat, st.session_state.lon),
            popup="Posição atual",
            draggable=True,
            icon=folium.Icon(color="blue", icon="map-marker")
        ).add_to(fmap)
        st.markdown("**Clique no mapa** para posicionar a observação (ou arraste o marcador).")
        map_state: Dict[str, Any] = st_folium(fmap, height=420, width=None, returned_objects=["last_clicked", "last_object"])
        if map_state and map_state.get("last_clicked"):
            lat = float(map_state["last_clicked"]["lat"])
            lon = float(map_state["last_clicked"]["lng"])
            st.session_state.lat, st.session_state.lon = clamp_to_bounds(lat, lon)
        if map_state and map_state.get("last_object"):
            obj = map_state["last_object"]
            if obj and obj.get("type") == "Feature" and obj.get("geometry", {}).get("type") == "Point":
                coords = obj["geometry"]["coordinates"]
                lat, lon = float(coords[1]), float(coords[0])
                st.session_state.lat, st.session_state.lon = clamp_to_bounds(lat, lon)

    elif input_mode == "Digitar coordenadas":
        c1, c2 = st.columns(2)
        with c1:
            lat_in = st.number_input("Latitude", value=st.session_state.get("lat", -22.90), step=0.0001, format="%.5f")
        with c2:
            lon_in = st.number_input("Longitude", value=st.session_state.get("lon", -43.20), step=0.0001, format="%.5f")
        st.session_state.lat, st.session_state.lon = clamp_to_bounds(float(lat_in), float(lon_in))

    elif input_mode == "Buscar por cidade":
        st.markdown("Digite o nome da cidade e estado (ex: `Teresópolis, RJ`) e clique em buscar.")
        search_c1, search_c2 = st.columns([0.75, 0.25])
        with search_c1:
            city_query = st.text_input("Cidade, Estado", "Teresópolis, RJ", label_visibility="collapsed")
        with search_c2:
            search_btn = st.button("Buscar", use_container_width=True)
        if search_btn and city_query.strip():
            coords = geocode_city(city_query.strip())
            if coords:
                lat, lon = coords
                st.session_state.lat, st.session_state.lon = clamp_to_bounds(lat, lon)
                st.success(f"Localização encontrada para '{city_query}'!")
            else:
                st.error(f"Não foi possível encontrar a localização para '{city_query}'. Tente ser mais específico.")

    st.caption(nice_location_label(st.session_state.lat, st.session_state.lon))
    st.write("")
    run_btn = st.button("Rodar inferência", type="primary", use_container_width=True)

with col_right:
    st.subheader("Prévia da imagem")
    try:
        if image_url.strip():
            img = load_pil_from_url(image_url.strip())
            st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível carregar a imagem: {e}")

st.markdown("---")

if run_btn:
    try:
        result = predictor.predict(
            image=image_url.strip(),
            observed_on=date_str.strip(),
            latitude=float(st.session_state.lat),
            longitude=float(st.session_state.lon),
            return_intermediate=True,
        )
        proba = float(result["prob_H"])
        label = result["label"]
        thr = float(result["threshold"])

        st.subheader("Resultado da Classificação")
        c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
        with c1:
            st.metric("Palpite", label)
        with c2:
            st.metric("Prob. (classe H)", f"{proba:.3f}")
        with c3:
            st.metric("Limiar usado", f"{thr:.3f}")

        with st.expander("Detalhes técnicos"):
            st.json(result)

    except Exception as e:
        st.error(f"Falha na inferência: {e}")