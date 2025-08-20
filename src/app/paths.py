from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
CHECKPOINTS_DIR = PROJ_ROOT / "checkpoints"
OUTPUTS_DIR = PROJ_ROOT / "outputs"

IMAGES_CACHE = DATA_DIR / "images_cache"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
MODELS_DIR = OUTPUTS_DIR / "models"

for p in [DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, IMAGES_CACHE, EMBEDDINGS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
