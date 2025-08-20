# src/app/data/images.py
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from app.utils.hash import sha1_hex
from app.paths import IMAGES_CACHE

# sessão global com pool + retry
_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=64, pool_maxsize=64,
    max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    ),
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update({"User-Agent": "saguis-dinov3/0.1"})

def fetch_image_to_cache(url: str) -> Path:
    name = sha1_hex(url) + ".jpg"
    out = IMAGES_CACHE / name
    if not out.exists():
        r = _session.get(url, timeout=(5, 30))
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="JPEG", quality=92, optimize=True)
    return out

def load_pil_from_url(url: str):
    """
    Retorna PIL.Image em caso de sucesso; caso contrário, retorna None.
    """
    try:
        path = fetch_image_to_cache(url)
        return Image.open(path).convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError):
        return None
