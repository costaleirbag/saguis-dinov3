# streamlit_app.py — com diagnóstico de paridade e opção de resize 224

import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
from PIL import Image
import requests
from ultralytics import YOLO
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
# Constantes
# =========================
IBGE_URBAN_AREAS_PATH = "data/geo/ibge_areas_urbanizadas.gpkg"
IBGE_URBAN_AREAS_LAYER = "lml_area_densamente_edificada_a"

SE_BOUNDS = {"min_lat": -25.5, "max_lat": -14.0, "min_lon": -53.0, "max_lon": -39.0}
SE_CENTER = (-21.0, -44.0)
SE_ZOOM = 5

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
# Helpers
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
        headers = {"User-Agent": "Sagui-DINOv3-Streamlit/1.0 (contato@exemplo.com)"}
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d:
            return float(d[0]["lat"]), float(d[0]["lon"])
        return None
    except Exception as e:
        st.error(f"Erro de geocodificação: {e}")
        return None

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

def parse_date_to_iso(date_str: str) -> str:
    try:
        return pd.to_datetime(date_str.strip(), dayfirst=True).date().isoformat()
    except Exception:
        return date_str.strip()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# =========================
# Carregamento de Modelos
# =========================
@st.cache_resource(show_spinner="A carregar o classificador...")
def load_predictor(artifacts_path: str, hf_model: str, device_prefer: str):
    cfg = PredictorConfig(
        artifacts_path=artifacts_path,
        hf_model=hf_model,
        device_prefer=device_prefer,
    )
    return SaguiPredictor(cfg)

@st.cache_resource(show_spinner="A carregar o YOLO...")
def load_yolo_model(yolo_model_name: str):
    return YOLO(yolo_model_name)

@st.cache_resource(show_spinner="A carregar dados do IBGE...")
def load_ibge_urban_areas_cached():
    try:
        if gpd is None:
            raise ImportError("Instale: geopandas shapely pyproj")
        return gpd.read_file(IBGE_URBAN_AREAS_PATH, layer=IBGE_URBAN_AREAS_LAYER)
    except Exception as e:
        st.error(f"Não foi possível carregar o IBGE: {e}")
        return None

# =========================
# UI
# =========================
st.set_page_config(page_title="Saguis DINOv3 • Demo", layout="wide")
st.title("Saguis (H vs N-H) • DINOv3 + LightGBM + PCA")

with st.sidebar:
    st.header("Modelo")
    artifacts_path = st.text_input("Artefatos (.joblib)", "outputs/tests/urban_geopy/final_model.joblib")
    hf_model = st.text_input("DINOv3 (HF)", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    device_prefer = st.selectbox("Dispositivo", ["mps", "cuda", "cpu"], index=0)

    st.header("YOLO (visualização/crop)")
    yolo_model_name = st.selectbox("YOLOv8", ["yolov8l.pt","yolov8m.pt","yolov8s.pt"], index=0)
    yolo_device = st.selectbox("Device YOLO", ["mps","cuda","cpu"], index=0)
    yolo_imgsz = st.slider("imgsz (1ª)", 640, 1536, 960, 64)
    yolo_conf_fast = st.slider("conf (1ª)", 0.05, 0.80, 0.25, 0.01)
    yolo_iou_fast = st.slider("iou (1ª)", 0.30, 0.80, 0.55, 0.01)
    yolo_max_det = st.slider("max_det", 1, 20, 7, 1)
    yolo_expand = st.slider("expand", 0.00, 0.40, 0.20, 0.01)
    yolo_min_area_ratio = st.slider("min_area_ratio", 0.0, 0.20, 0.02, 0.005)
    yolo_min_side_px = st.slider("min_side_px", 8, 128, 16, 2)
    yolo_max_ar = st.slider("max AR", 2.0, 10.0, 6.0, 0.5)
    box_select_mode = st.selectbox("Seleção de caixa", ["area","conf","conf_area"], index=0)

    st.caption("Fallback (2ª passada)")
    yolo_conf_fb = st.slider("conf (fb)", 0.03, 0.50, max(0.05, yolo_conf_fast*0.6), 0.01)
    yolo_iou_fb = st.slider("iou (fb)", 0.40, 0.80, min(0.70, yolo_iou_fast+0.10), 0.01)
    yolo_imgsz_fb= st.slider("imgsz (fb)", 640, 1536, max(1280, yolo_imgsz), 64)

    use_coco_animals = st.checkbox("Restringir a COCO animais", True)
    classes_arg = st.text_input("Classes COCO (nome ou índice)", "")

    st.markdown("---")
    use_yolo_crop_for_pred = st.checkbox("Usar recorte YOLO na classificação", True)
    force_square_224 = st.checkbox("Forçar resize 224×224 antes do embedding", False,
                                   help="Mitiga diferença entre HF (center-crop) e treino (resize quadrado).")

# Carregar modelos
try:
    predictor = load_predictor(artifacts_path, hf_model, device_prefer)
    yolo_model = load_yolo_model(yolo_model_name)
except Exception as e:
    st.error(f"Falha ao carregar modelos: {e}")
    st.stop()

# Resolver classes
try:
    classes = parse_classes_arg(classes_arg)
except ValueError as e:
    st.error(str(e))
    st.stop()
if use_coco_animals:
    animal_idxs = [COCO80.index(n) for n in COCO_ANIMAL_NAMES]
    classes = sorted(set(animal_idxs if classes is None else classes + animal_idxs))

# Layout
col_left, col_right = st.columns([0.6, 0.4], gap="large")

with col_left:
    st.subheader("Entrada")
    image_url = st.text_input("URL da imagem", "https://inaturalist-open-data.s3.amazonaws.com/photos/199431875/medium.jpeg")
    date_str = st.text_input("Data do registo", "14/06/2022", help="Aceita DD/MM/AAAA ou AAAA-MM-DD")
    input_mode = st.radio("Localização", ["Clicar no mapa", "Digitar coordenadas", "Procurar por cidade"], horizontal=True)

    if "lat" not in st.session_state:
        st.session_state.lat, st.session_state.lon = -23.5438686236, -46.760838914

    if input_mode == "Clicar no mapa":
        fmap = folium.Map(location=(st.session_state.lat, st.session_state.lon), zoom_start=SE_ZOOM, tiles="OpenStreetMap")
        folium.Marker((st.session_state.lat, st.session_state.lon), popup="Posição atual",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(fmap)
        map_state = st_folium(fmap, height=420, width=None, key="se_map", returned_objects=["last_clicked"])
        if map_state and map_state.get("last_clicked"):
            lat = float(map_state["last_clicked"]["lat"])
            lon = float(map_state["last_clicked"]["lng"])
            st.session_state.lat, st.session_state.lon = clamp_to_bounds(lat, lon)
    elif input_mode == "Digitar coordenadas":
        c1, c2 = st.columns(2)
        lat_in = c1.number_input("Latitude", value=st.session_state.lat, step=0.0001, format="%.5f")
        lon_in = c2.number_input("Longitude", value=st.session_state.lon, step=0.0001, format="%.5f")
        st.session_state.lat, st.session_state.lon = clamp_to_bounds(float(lat_in), float(lon_in))
    else:
        city_query = st.text_input("Cidade, Estado", "Teresópolis, RJ")
        if st.button("Procurar"):
            coords = geocode_city(city_query)
            if coords: st.session_state.lat, st.session_state.lon = clamp_to_bounds(coords[0], coords[1])

    st.caption(nice_location_label(st.session_state.lat, st.session_state.lon))
    st.write("")
    run_btn = st.button("Rodar inferência", type="primary", use_container_width=True)

with col_right:
    st.subheader("Prévia / Recorte YOLO")
    original_image = None
    crop_info = None
    if image_url.strip():
        try:
            original_image = load_pil_from_url(image_url.strip())
            st.image(original_image, caption="Imagem Original", use_container_width=True)
            crop_info = process_image(
                image_url=image_url.strip(), model=yolo_model, device=yolo_device,
                conf=yolo_conf_fast, iou=yolo_iou_fast, imgsz=yolo_imgsz, max_det=yolo_max_det,
                conf_fb=yolo_conf_fb, iou_fb=yolo_iou_fb, imgsz_fb=yolo_imgsz_fb,
                min_area_ratio=yolo_min_area_ratio, min_side_px=yolo_min_side_px,
                max_ar=yolo_max_ar, expand=yolo_expand, classes=classes, box_select_mode=box_select_mode,
            )
            if crop_info:
                st.image(crop_info["crop"], caption=f"Crop YOLO • conf={crop_info['conf']:.2f} • cls={crop_info['cls_name'] or crop_info['cls']}", use_container_width=True)
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
            st.warning(f"Falha ao processar a imagem: {e}")

st.markdown("---")

# -------------------------
# Classificação + Diagnóstico
# -------------------------
if run_btn:
    st.session_state.show_results = True

if st.session_state.get("show_results", False):
    try:
        # IBGE
        ibge_gdf = load_ibge_urban_areas_cached()
        if ibge_gdf is None:
            st.stop()

        # ponto do usuário
        user_point_gdf = gpd.GeoDataFrame(
            {'name': ['Ponto de Avaliação']},
            geometry=[Point(st.session_state.lon, st.session_state.lat)],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        ibge_gdf_m = ibge_gdf.to_crs("EPSG:3857")
        nearest_urban_sjoin = gpd.sjoin_nearest(user_point_gdf, ibge_gdf_m, how="left", distance_col="distance_m")

        # Escolher imagem para predição
        img_for_pred = crop_info["crop"] if (use_yolo_crop_for_pred and crop_info and crop_info.get("crop")) else original_image
        if img_for_pred is None:
            st.error("Sem imagem válida para predição.")
            st.stop()

        # 🔧 Forçar resize 224×224 (mitigação de pré-processo)
        if force_square_224:
            try:
                img_for_pred = img_for_pred.resize((224, 224), Image.BICUBIC)
            except Exception:
                pass

        observed_iso = parse_date_to_iso(date_str)

        # ---------- Diagnóstico: extrair embedding bruto também ----------
        # Vamos pegar o embedding ANTES dos pipelines (igual notebook faz).
        # O SaguiPredictor não expõe método público, então usamos a imagem
        # e depois medimos a forma pós-embedding_pipeline para log.
        # (O embedding bruto é acessível via um helper rápido abaixo.)

        @st.cache_data(show_spinner=False)
        def _embed_once(_img_bytes: bytes, _artifacts_path: str, _hf: str, _dev: str, _force224: bool):
            # recria um predictor leve e gera embedding bruto uma vez
            cfg = PredictorConfig(artifacts_path=_artifacts_path, hf_model=_hf, device_prefer=_dev)
            _pred = SaguiPredictor(cfg)
            from io import BytesIO
            pil = Image.open(BytesIO(_img_bytes)).convert("RGB")
            if _force224:
                pil = pil.resize((224,224), Image.BICUBIC)
            # usa método privado do predictor via o extractor público
            emb = _pred.extractor.embed_pils_batch([pil])[0]  # np.ndarray (D,)
            return emb.astype(np.float32)

        # materializar bytes para cache estável
        img_bytes = io.BytesIO()
        img_for_pred.save(img_bytes, format="PNG")
        raw_emb = _embed_once(img_bytes.getvalue(), artifacts_path, hf_model, device_prefer, force_square_224)

        # Predição normal (com pipelines)
        result = predictor.predict(
            image=img_for_pred,
            observed_on=observed_iso,
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
        c3.metric("Limiar", f"{thr:.3f}")

        # ---------------- Mapa de área urbana ----------------
        st.subheader("Área Urbana Próxima")
        final_map = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=12, tiles="OpenStreetMap")
        folium.Marker((st.session_state.lat, st.session_state.lon), popup="Ponto de Avaliação",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(final_map)
        if not nearest_urban_sjoin.empty and not nearest_urban_sjoin['index_right'].isna().all():
            matched_idx = nearest_urban_sjoin['index_right'].iloc[0]
            distance_m = nearest_urban_sjoin["distance_m"].iloc[0]
            nearest_poly = ibge_gdf.loc[[matched_idx]]
            folium.GeoJson(
                nearest_poly,
                style_function=lambda x: {
                    "fillColor": "blue" if distance_m < 10 else "gray",
                    "color": "blue" if distance_m < 10 else "gray",
                    "weight": 2, "fillOpacity": 0.4
                },
                tooltip=folium.GeoJsonTooltip(fields=["nome"], aliases=["Nome"])
            ).add_to(final_map)
            if distance_m < 10:
                st.info("Ponto **dentro** de área urbana.")
            else:
                st.info(f"Ponto a **{distance_m/1000.0:.2f} km** da área urbana mais próxima.")
            centroid_wgs84 = nearest_poly.geometry.centroid.to_crs("EPSG:4326").iloc[0]
            folium.PolyLine([(st.session_state.lat, st.session_state.lon), (centroid_wgs84.y, centroid_wgs84.x)],
                            color='gray', weight=1, dash_array='5,5').add_to(final_map)
        st_folium(final_map, height=480, width=None)

        # ---------------- Diagnóstico de Paridade ----------------
        st.subheader("Diagnóstico de Paridade (Notebook × App)")
        with st.expander("Comparar embedding do app com um embedding de referência do notebook"):
            st.caption("Cole abaixo um vetor (JSON, CSV de uma linha, ou .npy) gerado **no notebook** para a MESMA imagem/crop.")
            ref_txt = st.text_area("Vetor de referência (opção 1: JSON/CSV de uma linha)", height=120)
            ref_file = st.file_uploader("ou carregue um .npy (opção 2)", type=["npy"], accept_multiple_files=False)

            ref_vec = None
            if ref_file is not None:
                try:
                    ref_vec = np.load(ref_file)
                except Exception as e:
                    st.error(f"Erro lendo .npy: {e}")
            elif ref_txt.strip():
                try:
                    # tenta JSON
                    ref_vec = np.array(json.loads(ref_txt), dtype=np.float32)
                except Exception:
                    try:
                        # tenta CSV simples
                        ref_vec = np.fromstring(ref_txt.strip(), sep=',', dtype=np.float32)
                    except Exception as e:
                        st.error(f"Não consegui interpretar o texto como vetor: {e}")

            if ref_vec is not None:
                if ref_vec.ndim > 1:
                    ref_vec = ref_vec.ravel()
                if raw_emb.shape != ref_vec.shape:
                    st.warning(f"Dimensões diferentes: app {raw_emb.shape} vs ref {ref_vec.shape}.")
                sim = cosine_similarity(raw_emb, ref_vec)
                st.metric("Similaridade cosseno (embeddings brutos)", f"{sim:.4f}",
                          help="≥ 0.95 costuma indicar pré-processos equivalentes.")
                if sim < 0.95:
                    st.warning("⚠️ Baixa similaridade → provável diferença de pré-processamento (HF vs square-resize) "
                               "ou recorte YOLO não idêntico. Ative o 'Forçar resize 224×224' e compare novamente.")

        # ---------------- Detalhes do Modelo / Importâncias ----------------
        st.subheader("Detalhes do Modelo")
        results_file_path = artifacts_path  # o próprio .joblib pode carregar os detalhes
        try:
            model_details = joblib.load(results_file_path)
        except Exception:
            model_details = {}

        with st.expander("Importâncias de features"):
            if isinstance(model_details.get('feature_importances'), list):
                df_features = pd.DataFrame(model_details['feature_importances']).rename(
                    columns={'feature': 'Feature', 'importance': 'Importance'}
                ).sort_values('Importance', ascending=False)
                def fmt_name(n): return 'embedding' if str(n).startswith('emb_') else n
                df_features['Tipo'] = df_features['Feature'].apply(fmt_name)
                df_grouped = df_features.groupby('Tipo')['Importance'].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(df_grouped, x='Tipo', y='Importance',
                             title='Importância total por tipo', labels={'Tipo':'Tipo', 'Importance':'Importância'})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_features.head(20).style.format({"Importance": "{:.5f}"}), use_container_width=True)
            else:
                st.info("Sem bloco 'feature_importances' nos artefatos.")

        # ---------------- Log técnico (para bater com o notebook) ----------------
        with st.expander("Log técnico (para checagem de paridade)"):
            st.code(
                f"""
observed_on (ISO): {observed_iso}
Embedding bruto: shape={raw_emb.shape}, dtype={raw_emb.dtype}
Embedding norm L2: {np.linalg.norm(raw_emb):.6f}
Pipelines:
  - embedding_pipeline: {type(predictor.embedding_pipeline).__name__}
  - tabular_pipeline  : {type(predictor.tabular_pipeline).__name__}
Modelo: {type(predictor.model).__name__}
""",
                language="text"
            )
            # gera tab_features iguais às do predictor para inspecionar
            from app.features.tabular import engineer_tab_features
            tab_cfg = predictor.tabular_config
            df_input = pd.DataFrame([{
                "observed_on": observed_iso,
                "latitude": float(st.session_state.lat),
                "longitude": float(st.session_state.lon),
            }])
            tab_df, _ = engineer_tab_features(
                df_input, mode="latlon_time",
                urban_areas_path=tab_cfg.get("urban_areas_path", IBGE_URBAN_AREAS_PATH),
                urban_layer=tab_cfg.get("urban_layer", IBGE_URBAN_AREAS_LAYER),
                urban_radius_km=tab_cfg.get("urban_radius_km", 5.0),
            )
            # reordenar conforme treino (para comparar no notebook)
            try:
                tab_df = tab_df[predictor.tabular_feature_names]
            except Exception:
                pass
            st.write("Colunas tabulares usadas (ordem do treino):", list(tab_df.columns))
            st.dataframe(tab_df, use_container_width=True)

    except Exception as e:
        st.error(f"Falha na inferência: {e}")
        st.exception(e)
