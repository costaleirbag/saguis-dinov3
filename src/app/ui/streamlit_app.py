import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
from PIL import Image, ImageOps
import requests
from ultralytics import YOLO
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from app.inference.predictor import SaguiPredictor, PredictorConfig
from app.pipeline.preprocess_and_filter_images import process_image
from app.data.images import load_pil_from_url

import geopandas as gpd
from shapely.geometry import Point
import folium.plugins
import json

# =========================
# Constantes Adicionais
# =========================
IBGE_URBAN_AREAS_PATH = "data/geo/ibge_areas_urbanizadas.gpkg"
IBGE_URBAN_AREAS_LAYER = "lml_area_densamente_edificada_a"

# =========================
# Config e Constantes
# =========================
SE_BOUNDS = {
    "min_lat": -25.5, "max_lat": -14.0,
    "min_lon": -53.0, "max_lon": -39.0
}
SE_CENTER = (-21.0, -44.0)
SE_ZOOM = 5
YOLO_CONF_THRESHOLD = 0.5  # valor padrão; será sobrescrito pelos sliders

# COCO80 (Ultralytics/YOLO)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
COCO_ANIMAL_NAMES = ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]


# =========================
# Helpers gerais
# =========================
def clamp_to_bounds(lat: float, lon: float) -> Tuple[float, float]:
    lat = max(min(lat, SE_BOUNDS["max_lat"]), SE_BOUNDS["min_lat"])
    lon = max(min(lon, SE_BOUNDS["max_lon"]), SE_BOUNDS["min_lon"])
    return lat, lon

def se_contains(lat: float, lon: float) -> bool:
    return (SE_BOUNDS["min_lat"] <= lat <= SE_BOUNDS["max_lat"]) and \
           (SE_BOUNDS["min_lon"] <= lon <= SE_BOUNDS["max_lon"])

def nice_location_label(lat: float, lon: float) -> str:
    in_se = se_contains(lat, lon)
    return f"Lat {lat:.5f}, Lon {lon:.5f}" + (" • Sudeste do Brasil" if in_se else "")

@st.cache_data(show_spinner="A procurar localização...")
def geocode_city(query: str) -> Optional[Tuple[float, float]]:
    try:
        headers = {"User-Agent": "Sagui-DINOv3-Streamlit-Demo/1.0 (seu-email@exemplo.com)"}
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            return lat, lon
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com o serviço de geolocalização: {e}")
        return None
    except (KeyError, IndexError):
        return None

def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

def expand_and_clamp_bbox(bbox: Tuple[float,float,float,float], w: int, h: int, expand: float):
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    nx, ny = bw*(1+expand), bh*(1+expand)
    x1n, y1n = cx-nx/2, cy-ny/2
    x2n, y2n = cx+nx/2, cy+ny/2
    return (clamp(x1n,0,w-1), clamp(y1n,0,h-1), clamp(x2n,0,w-1), clamp(y2n,0,h-1))

def choose_box(confs: List[float], areas: List[float], mode: str = "conf_area") -> int:
    if mode == "conf":
        return int(np.argmax(confs))
    if mode == "area":
        return int(np.argmax(areas))
    score = np.array(confs) * np.array(areas)
    return int(np.argmax(score))

def parse_classes_arg(arg: str) -> Optional[List[int]]:
    if not arg or not arg.strip():
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    idxs: List[int] = []
    lower = [n.lower() for n in COCO80]
    for p in parts:
        if p.isdigit():
            idxs.append(int(p))
        else:
            name = p.lower()
            if name not in lower:
                raise ValueError(f"Classe '{p}' não existe no COCO80.")
            idxs.append(lower.index(name))
    return idxs


# =========================
# Carregamento de Modelos
# =========================
@st.cache_resource(show_spinner="A carregar o modelo de classificação...")
def load_predictor(artifacts_path: str, hf_model: str, device_prefer: str):
    cfg = PredictorConfig(
        artifacts_path=artifacts_path,
        hf_model=hf_model,
        device_prefer=device_prefer,
    )
    return SaguiPredictor(cfg)

@st.cache_resource(show_spinner="A carregar o modelo YOLO...")
def load_yolo_model(yolo_model_name: str):
    return YOLO(yolo_model_name)

@st.cache_resource(show_spinner="A carregar dados do IBGE...")
def load_ibge_urban_areas_cached():
    """
    Carrega o GeoDataFrame do IBGE apenas uma vez.
    """
    try:
        if gpd is None:
             raise ImportError("Geospatial dependencies missing. Run: poetry add geopandas shapely pyproj")
        
        return gpd.read_file(IBGE_URBAN_AREAS_PATH, layer=IBGE_URBAN_AREAS_LAYER)
    except Exception as e:
        st.error(f"Não foi possível carregar o arquivo do IBGE: {e}")
        return None


# =========================
# UI
# =========================
st.set_page_config(page_title="Saguis DINOv3 - Demo", layout="wide")
st.title("Saguis (H vs N-H) • DINOv3 + LightGBM + PCA")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuração do Modelo")
    artifacts_path = st.text_input(
        "Caminho do ficheiro de artefactos do modelo (.joblib)",
        "outputs/tests/urban_geopy/final_model.joblib"
    )
    hf_model = st.text_input(
        "Modelo DINOv3 (Hugging Face)",
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    device_prefer = st.selectbox(
        "Dispositivo (classificador)",
        ["mps", "cuda", "cpu"],
        help="Selecione 'cuda' para Nvidia, 'mps' para Mac Apple Silicon, ou 'cpu'."
    )

    st.header("YOLO • Modelo e Parâmetros (Visualização)")
    yolo_model_name = st.selectbox(
        "Modelo YOLOv8",
        ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt"],
        index=0
    )
    yolo_device = st.selectbox("Device YOLO", ["mps","cuda","cpu"], index=0)
    yolo_imgsz = st.slider("imgsz (1ª passada)", 640, 1536, 960, 64)
    yolo_conf_fast = st.slider("conf (1ª passada)", 0.05, 0.80, 0.25, 0.01)
    yolo_iou_fast  = st.slider("iou (1ª passada)",  0.30, 0.80, 0.55, 0.01)
    yolo_max_det   = st.slider("max_det", 1, 20, 7, 1)
    yolo_expand    = st.slider("expand (margem do crop)", 0.00, 0.40, 0.20, 0.01)
    yolo_min_area_ratio = st.slider("min_area_ratio", 0.0, 0.20, 0.02, 0.005)
    yolo_min_side_px    = st.slider("min_side_px", 8, 128, 16, 2)
    yolo_max_ar         = st.slider("max aspect ratio", 2.0, 10.0, 6.0, 0.5)
    box_select_mode     = st.selectbox("Seleção de caixa", ["area","conf","conf_area"], index=0)

    st.caption("Fallback (2ª passada) mais indulgente")
    yolo_conf_fb = st.slider("conf (fallback)", 0.03, 0.50, max(0.05, yolo_conf_fast*0.6), 0.01)
    yolo_iou_fb  = st.slider("iou (fallback)",  0.40, 0.80, min(0.70, yolo_iou_fast+0.10), 0.01)
    yolo_imgsz_fb= st.slider("imgsz (fallback)", 640, 1536, max(1280, yolo_imgsz), 64)

    use_coco_animals = st.checkbox("Restringir a COCO animais", True)
    classes_arg = st.text_input("Classes COCO (ex.: cat,dog ou 15,16)", "")

# --- Carregamento dos modelos ---
try:
    predictor = load_predictor(artifacts_path, hf_model, device_prefer)
    yolo_model = load_yolo_model(yolo_model_name)
except Exception as e:
    st.error(f"Falha ao carregar os modelos: {e}")
    st.stop()

# Resolver classes COCO para a visualização
try:
    classes = parse_classes_arg(classes_arg)
except ValueError as e:
    st.error(str(e))
    st.stop()
if use_coco_animals:
    animal_idxs = [COCO80.index(n) for n in COCO_ANIMAL_NAMES]
    classes = sorted(set(animal_idxs if classes is None else classes + animal_idxs))

# --- Interface Principal ---
col_left, col_right = st.columns([0.6, 0.4], gap="large")

with col_left:
    st.subheader("Entrada")
    image_url = st.text_input("URL da imagem", "https://inaturalist-open-data.s3.amazonaws.com/photos/59806858/medium.png")
    date_str = st.text_input("Data do registo", "23/10/2009", help="Aceita DD/MM/AAAA ou AAAA-MM-DD")

    input_mode = st.radio(
        "Como informar a localização?",
        ["Clicar no mapa", "Digitar coordenadas", "Procurar por cidade"],
        horizontal=True
    )

    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lon = -22.90, -43.20

    if input_mode == "Clicar no mapa":
        # Cria o mapa sempre centrado no último estado salvo
        fmap = folium.Map(
            location=(st.session_state.lat, st.session_state.lon),
            zoom_start=SE_ZOOM,
            control_scale=True,
            tiles="OpenStreetMap"
        )

        # marcador somente para visualização do ponto atual (NÃO draggable)
        folium.Marker(
            location=(st.session_state.lat, st.session_state.lon),
            popup="Posição atual",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(fmap)

        # retorna só o que vamos realmente usar; dê uma key fixa
        map_state = st_folium(
            fmap,
            height=420,
            width=None,
            key="se_map",
            returned_objects=["last_clicked"]  # << importante
        )

        # Atualiza sessão quando houver clique novo
        if map_state and map_state.get("last_clicked"):
            lat = float(map_state["last_clicked"]["lat"])
            lon = float(map_state["last_clicked"]["lng"])
            st.session_state.lat, st.session_state.lon = clamp_to_bounds(lat, lon)
    elif input_mode == "Digitar coordenadas":
        c1, c2 = st.columns(2)
        lat_in = c1.number_input("Latitude", value=st.session_state.lat, step=0.0001, format="%.5f")
        lon_in = c2.number_input("Longitude", value=st.session_state.lon, step=0.0001, format="%.5f")
        st.session_state.lat, st.session_state.lon = clamp_to_bounds(float(lat_in), float(lon_in))
    elif input_mode == "Procurar por cidade":
        city_query = st.text_input("Cidade, Estado", "Teresópolis, RJ")
        if st.button("Procurar"):
            coords = geocode_city(city_query)
            if coords:
                st.session_state.lat, st.session_state.lon = clamp_to_bounds(coords[0], coords[1])

    st.caption(nice_location_label(st.session_state.lat, st.session_state.lon))
    st.write("")
    run_btn = st.button("Rodar inferência", type="primary", use_container_width=True)

with col_right:
    st.subheader("Prévia da Imagem")
    if image_url.strip():
        try:
            original_image = load_pil_from_url(image_url.strip())
            st.image(original_image, caption="Imagem Original", use_container_width=True)

            # --- Recorte YOLO (usa a mesma função do pipeline) ---
            crop_info = process_image(
                image_url=image_url.strip(),
                model=yolo_model,
                device=yolo_device,
                conf=yolo_conf_fast,
                iou=yolo_iou_fast,
                imgsz=yolo_imgsz,
                max_det=yolo_max_det,
                conf_fb=yolo_conf_fb,
                iou_fb=yolo_iou_fb,
                imgsz_fb=yolo_imgsz_fb,
                min_area_ratio=yolo_min_area_ratio,
                min_side_px=yolo_min_side_px,
                max_ar=yolo_max_ar,
                expand=yolo_expand,
                classes=classes,
                box_select_mode=box_select_mode,
            )

            if crop_info:
                st.image(
                    crop_info["crop"],
                    caption=f"Recorte YOLO • conf={crop_info['conf']:.2f} • classe={crop_info['cls_name'] or crop_info['cls']}",
                    use_container_width=True
                )
                with st.expander("Detalhes do recorte"):
                    st.json({
                        "bbox_raw_xyxy": crop_info["bbox_raw_xyxy"],
                        "bbox_expanded_xyxy": crop_info["bbox_xyxy"],
                        "area_ratio": round(crop_info["area_ratio"], 5),
                        "params": crop_info["params"] | {"model": yolo_model_name, "device": yolo_device}
                    })
            else:
                st.info("Nenhuma caixa válida após filtros.")
        except Exception as e:
            st.warning(f"Não foi possível carregar ou processar a imagem: {e}")

st.markdown("---")

# --- Classificação e visualização (VERSÃO FINAL E CORRIGIDA) ---
# Lógica de controle do estado para manter os resultados na tela
if run_btn:
    st.session_state.show_results = True

if st.session_state.get('show_results', False):
    try:
        # Carregue o GeoDataFrame do IBGE (usando a função cacheada)
        ibge_gdf = load_ibge_urban_areas_cached()
        if ibge_gdf is None:
            st.stop()
            
        # Crie um GeoDataFrame para o ponto do usuário
        user_point_gdf = gpd.GeoDataFrame(
            {'name': ['Ponto de Avaliação']}, 
            geometry=[Point(st.session_state.lon, st.session_state.lat)], 
            crs="EPSG:4326"
        )

        # Use um CRS métrico para o cálculo de distância
        user_point_gdf_m = user_point_gdf.to_crs("EPSG:3857")
        ibge_gdf_m = ibge_gdf.to_crs("EPSG:3857")
        
        # Encontre o polígono urbano mais próximo
        nearest_urban_sjoin = gpd.sjoin_nearest(
            user_point_gdf_m,
            ibge_gdf_m,
            how="left",
            distance_col="distance_m"
        )
        
        # O resto do seu código de inferência
        result = predictor.predict(
            image=image_url.strip(),
            observed_on=date_str.strip(),
            latitude=float(st.session_state.lat),
            longitude=float(st.session_state.lon),
        )
        
        proba = float(result["prob_H"])
        label = result["label"]
        thr = float(result["threshold"])

        st.subheader("Resultado da Classificação")
        c1, c2, c3 = st.columns(3)
        c1.metric("Palpite", label)
        c2.metric("Prob. (classe H)", f"{proba:.3f}")
        c3.metric("Limiar usado", f"{thr:.3f}")
        
        # ==========================================================
        # Seção de Plotagem da Área Urbana
        # ==========================================================
        st.subheader("Área Urbana Próxima")
        
        # Crie o mapa centrado no ponto do usuário
        final_map = folium.Map(
            location=[st.session_state.lat, st.session_state.lon], 
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        
        # Adicione o marcador do ponto avaliado
        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            popup="Ponto de Avaliação",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(final_map)

        # Verifique se uma junção foi encontrada e obtenha o polígono correto
        if not nearest_urban_sjoin.empty and not nearest_urban_sjoin['index_right'].isna().all():
            matched_idx = nearest_urban_sjoin['index_right'].iloc[0]
            # Use o índice para obter o GeoDataFrame do polígono correspondente do IBGE
            nearest_urban_polygon_gdf = ibge_gdf.loc[[matched_idx]]
            
            distance_m = nearest_urban_sjoin["distance_m"].iloc[0]

            # Use o GeoDataFrame corrigido para a plotagem
            folium.GeoJson(
                data=nearest_urban_polygon_gdf,
                style_function=lambda x: {
                    "fillColor": "blue" if distance_m < 10 else "gray",
                    "color": "blue" if distance_m < 10 else "gray",
                    "weight": 2,
                    "fillOpacity": 0.4
                },
                # Corrigido para usar os nomes de coluna reais do seu arquivo
                tooltip=folium.GeoJsonTooltip(fields=["nome"], aliases=["Nome"])
            ).add_to(final_map)

            if distance_m < 10:
                st.info("Ponto localizado **dentro** de uma área urbana.")
            else:
                dist_km = distance_m / 1000.0
                st.info(f"Ponto localizado a **{dist_km:.2f} km** da área urbana mais próxima.")
                
                # Adicione uma linha entre o ponto e o centróide do polígono
                centroid_wgs84 = nearest_urban_polygon_gdf.geometry.centroid.to_crs("EPSG:4326").iloc[0]
                # ATENÇÃO: Corrigido o nome da classe de PolyLine para Polyline
                folium.PolyLine(
                    locations=[(st.session_state.lat, st.session_state.lon), 
                            (centroid_wgs84.y, centroid_wgs84.x)],
                    color='gray',
                    weight=1,
                    dash_array='5, 5'
                ).add_to(final_map)

            # Renderize o mapa final
            st_folium(final_map, height=500, width=None)
        
        # --- Adicionar esta nova seção ---
        st.subheader("Detalhes do Modelo")
        
        # Carregar os resultados do arquivo joblib
        results_file_path = 'outputs/tests/urban_geopy/final_model.joblib'
        try:
            with open(results_file_path, 'rb') as f:
                model_details = joblib.load(f)
        except FileNotFoundError:
            st.error(f"Arquivo de modelo não encontrado: {results_file_path}")
            st.stop()
            
        with st.expander("Ver detalhes de features e importâncias"):
            # Verifica se 'feature_importances' existe e é uma lista (o que implica que contém os dicionários)
            if 'feature_importances' in model_details and isinstance(model_details['feature_importances'], list):
                feature_importances_data = model_details['feature_importances']

                # Cria um DataFrame diretamente a partir da lista de dicionários
                # As chaves 'feature' e 'importance' dos dicionários se tornarão as colunas do DataFrame
                df_features = pd.DataFrame(feature_importances_data)

                # Renomeia as colunas para 'Feature' e 'Importance' para consistência (opcional, mas bom para visualização)
                df_features.rename(columns={'feature': 'Feature', 'importance': 'Importance'}, inplace=True)

                # Ordena as features pela importância
                df_features = df_features.sort_values(by='Importance', ascending=False)

                # Formata o nome das features para agrupar os embeddings
                def format_feature_name(name):
                    if name.startswith('emb_'):
                        # Se for uma feature de embedding, retorna um nome genérico para agrupamento
                        return 'embedding'
                    # Caso contrário, retorna o nome da feature como está
                    return name

                df_features['Formatted_Feature'] = df_features['Feature'].apply(format_feature_name)

                # Agrupa por features formatadas e soma as importâncias
                df_grouped = df_features.groupby('Formatted_Feature')['Importance'].sum().sort_values(ascending=False).reset_index()

                st.write("### Importância das Features (agrupadas por tipo)")
                # Cria um gráfico de barras interativo usando Plotly
                fig_grouped = px.bar(df_grouped,
                                    x='Formatted_Feature',
                                    y='Importance',
                                    title='Importância Total por Tipo de Feature',
                                    labels={'Formatted_Feature': 'Tipo de Feature', 'Importance': 'Importância Total'},
                                    hover_data={'Formatted_Feature': True, 'Importance': ':.2f'}) # Formata o hover para 2 casas decimais

                # Personaliza o layout do gráfico
                fig_grouped.update_layout(xaxis_title="Tipo de Feature",
                                        yaxis_title="Importância Total",
                                        showlegend=False,
                                        uniformtext_minsize=8,
                                        uniformtext_mode='hide')

                st.plotly_chart(fig_grouped, use_container_width=True)


                st.write("### Lista completa de Features (Top 20)")
                # Exibe os resultados das top N features individuais
                st.dataframe(df_features.head(20).style.format({"Importance": "{:.5f}"}), use_container_width=True)

            else:
                st.warning("O arquivo de modelo não contém os dados de 'feature_importances' ou está em um formato inesperado.")
                
    except Exception as e:
        st.error(f"Falha na inferência: {e}")
        st.exception(e)