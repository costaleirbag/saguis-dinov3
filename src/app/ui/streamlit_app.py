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

from app.inference.predictor import SaguiPredictor, PredictorConfig
from app.data.images import load_pil_from_url

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


# =========================
# Função principal de recorte YOLO (espelha o pipeline)
# =========================
def yolo_preview_crop(
    yolo_model: YOLO,
    image: Image.Image,
    *,
    device: str = "mps",
    # fast-first-pass
    conf_fast: float = 0.25,
    iou_fast: float = 0.55,
    imgsz_fast: int = 960,
    max_det: int = 7,
    # fallback indulgente
    conf_fb: float = 0.10,
    iou_fb: float = 0.70,
    imgsz_fb: int = 1280,
    # filtros geométricos
    min_area_ratio: float = 0.02,
    min_side_px: int = 16,
    max_ar: float = 6.0,
    expand: float = 0.20,
    # classes COCO
    classes: Optional[List[int]] = None,
    box_select_mode: str = "area",
) -> Optional[Dict[str, Any]]:
    """Retorna dict com crop e metadados ou None se nada válido."""
    # 1ª passada (rápida)
    res1 = yolo_model.predict(
        image,
        verbose=False,
        device=device,
        conf=conf_fast,
        iou=iou_fast,
        max_det=max_det,
        classes=classes,
        imgsz=imgsz_fast
    )
    boxes = res1 and res1[0].boxes
    has_det = bool(boxes) and len(boxes) > 0

    # fallback indulgente
    if not has_det:
        res2 = yolo_model.predict(
            image,
            verbose=False,
            device=device,
            conf=conf_fb,
            iou=iou_fb,
            max_det=max(15, max_det),
            classes=classes,
            imgsz=imgsz_fb,
            # augment=True,  # se sua versão permitir TTA
        )
        boxes = res2 and res2[0].boxes
        has_det = bool(boxes) and len(boxes) > 0

    if not has_det:
        return None

    w, h = image.size
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().tolist()
    clss  = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []

    # filtros geométricos
    areas, keep = [], []
    for i,(x1,y1,x2,y2) in enumerate(xyxy):
        bw, bh = max(0, x2-x1), max(0, y2-y1)
        if bw < min_side_px or bh < min_side_px:
            continue
        big, small = (bw,bh) if bw>=bh else (bh,bw)
        if small==0 or (big/small) > max_ar:
            continue
        areas.append(bw*bh)
        keep.append(i)

    if not keep:
        return None

    confs_kept = [confs[i] for i in keep]
    areas_kept = [areas[j] for j in range(len(keep))]
    sel = choose_box(confs_kept, areas_kept, box_select_mode)
    idx = keep[sel]
    x1, y1, x2, y2 = xyxy[idx]
    area = max(1.0, (x2-x1)*(y2-y1))
    area_ratio = float(area/(w*h + 1e-9))

    if area_ratio < min_area_ratio:
        return None

    ex1, ey1, ex2, ey2 = expand_and_clamp_bbox((x1,y1,x2,y2), w, h, expand)
    crop_img = image.crop((ex1, ey1, ex2, ey2))
    cls_name = (COCO80[int(clss[idx])] if clss and 0 <= int(clss[idx]) < len(COCO80) else None)

    return dict(
        crop=crop_img,
        conf=float(confs[idx]),
        cls=int(clss[idx]) if clss else None,
        cls_name=cls_name,
        bbox_raw=[int(x1),int(y1),int(x2),int(y2)],
        bbox_expanded=[int(ex1),int(ey1),int(ex2),int(ey2)],
        area_ratio=area_ratio
    )


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
        "outputs/final_model_yolo_pca64/final_model_with_pca.joblib"
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
        ["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
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
        fmap = folium.Map(location=SE_CENTER, zoom_start=SE_ZOOM, control_scale=True, tiles="OpenStreetMap")
        folium.Marker(location=(st.session_state.lat, st.session_state.lon), popup="Posição atual", draggable=True).add_to(fmap)
        map_state = st_folium(fmap, height=420, width=None, returned_objects=["last_clicked"])
        if map_state and map_state.get("last_clicked"):
            lat, lon = map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"]
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

            # --- Recorte YOLO para visualização (função unificada) ---
            crop_info = yolo_preview_crop(
                yolo_model,
                original_image,
                device=yolo_device,
                conf_fast=yolo_conf_fast,
                iou_fast=yolo_iou_fast,
                imgsz_fast=yolo_imgsz,
                max_det=yolo_max_det,
                conf_fb=yolo_conf_fb,
                iou_fb=yolo_iou_fb,
                imgsz_fb=yolo_imgsz_fb,
                min_area_ratio=yolo_min_area_ratio,
                min_side_px=yolo_min_side_px,
                max_ar=yolo_max_ar,
                expand=yolo_expand,
                classes=classes,
                box_select_mode=box_select_mode
            )

            if crop_info:
                st.image(
                    crop_info["crop"],
                    caption=f"Recorte YOLO • conf={crop_info['conf']:.2f} • classe={crop_info['cls_name'] or crop_info['cls']}",
                    use_container_width=True
                )
                with st.expander("Detalhes do recorte"):
                    st.json({
                        "bbox_raw_xyxy": crop_info["bbox_raw"],
                        "bbox_expanded_xyxy": crop_info["bbox_expanded"],
                        "area_ratio": round(crop_info["area_ratio"], 5),
                        "params": {
                            "conf_fast": yolo_conf_fast, "iou_fast": yolo_iou_fast, "imgsz_fast": yolo_imgsz,
                            "conf_fb": yolo_conf_fb, "iou_fb": yolo_iou_fb, "imgsz_fb": yolo_imgsz_fb,
                            "max_det": yolo_max_det, "expand": yolo_expand,
                            "min_area_ratio": yolo_min_area_ratio, "min_side_px": yolo_min_side_px, "max_ar": yolo_max_ar,
                            "classes": classes, "use_coco_animals": use_coco_animals,
                            "model": yolo_model_name, "device": yolo_device
                        }
                    })
            else:
                st.info("Nenhuma caixa válida após filtros. "
                        "Tente diminuir min_area_ratio, min_side_px ou aumentar imgsz / fallback.")
        except Exception as e:
            st.warning(f"Não foi possível carregar ou processar a imagem: {e}")

st.markdown("---")

# --- Classificação (sua lógica original, sem mudanças) ---
if run_btn:
    try:
        result = predictor.predict(
            image=image_url.strip(),
            observed_on=date_str.strip(),
            latitude=float(st.session_state.lat),
            longitude=float(st.session_state.lon)
        )
        proba = float(result["prob_H"])
        label = result["label"]
        thr = float(result["threshold"])

        st.subheader("Resultado da Classificação")
        c1, c2, c3 = st.columns(3)
        c1.metric("Palpite", label)
        c2.metric("Prob. (classe H)", f"{proba:.3f}")
        c3.metric("Limiar usado", f"{thr:.3f}")

    except Exception as e:
        st.error(f"Falha na inferência: {e}")
        st.exception(e)
