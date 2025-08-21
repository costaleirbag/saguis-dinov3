import argparse
import json
from pathlib import Path
from glob import glob
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO

from app.data.images import load_pil_from_url
from app.utils.hash import sha1_hex

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque


# COCO80 (Ultralytics/YOLO) – nomes em ordem
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

# Whitelist “animais” do COCO (ajuda a não descartar se o sagui for rotulado como outro bicho)
COCO_ANIMAL_NAMES = [
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"
]


def parse_classes_arg(arg: str) -> Optional[List[int]]:
    """
    Converte "--classes" em lista de índices COCO.
    Aceita nomes (ex.: 'cat,dog') ou índices ('15,16').
    Retorna None se arg vazio.
    """
    if not arg or not arg.strip():
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    idxs: List[int] = []
    for p in parts:
        if p.isdigit():
            idxs.append(int(p))
        else:
            # nome -> índice (case insensitive)
            name = p.lower()
            if name not in [n.lower() for n in COCO80]:
                raise ValueError(f"Classe '{p}' não existe no COCO80.")
            idxs.append([n.lower() for n in COCO80].index(name))
    return idxs


def clamp(val, lo, hi):
    return max(lo, min(int(val), hi))


def expand_and_clamp_bbox(bbox, w, h, expand: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1), (y2 - y1)
    nx, ny = bw * (1 + expand), bh * (1 + expand)
    x1n, y1n = cx - nx / 2, cy - ny / 2
    x2n, y2n = cx + nx / 2, cy + ny / 2
    return (
        clamp(x1n, 0, w - 1),
        clamp(y1n, 0, h - 1),
        clamp(x2n, 0, w - 1),
        clamp(y2n, 0, h - 1),
    )


def choose_box(confs: List[float], areas: List[float], mode: str = "conf_area") -> int:
    # mode: "conf", "area", "conf_area"
    if mode == "conf":
        return int(np.argmax(confs))
    elif mode == "area":
        return int(np.argmax(areas))
    score = np.array(confs) * np.array(areas)
    return int(np.argmax(score))


def center_crop(pil: Image.Image, frac: float = 0.8) -> Image.Image:
    w, h = pil.size
    s = int(min(w, h) * frac)
    cx, cy = w // 2, h // 2
    x1 = clamp(cx - s // 2, 0, w - 1)
    y1 = clamp(cy - s // 2, 0, h - 1)
    x2 = clamp(cx + s // 2, 0, w - 1)
    y2 = clamp(cy + s // 2, 0, h - 1)
    return pil.crop((x1, y1, x2, y2))

def load_and_prepare_image(image_url: str):
    """Baixa, corrige EXIF e retorna PIL RGB + metadado (w,h)."""
    pil = load_pil_from_url(image_url)
    if pil is None:
        return None, None
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return pil, pil.size

def batched(iterable, n):
    """Gera listas de tamanho n (último pode ser menor)."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def prefetch_rows(rows_iter, prefetch: int, max_workers: int = 8):
    """
    Prefetch multi-thread de imagens.
    Produz (row, pil, (w,h)) na mesma ordem de chegada (não garante ordem original).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = deque()
        for row in rows_iter:
            fut = ex.submit(load_and_prepare_image, row.image_url)
            futures.append((row, fut))
            if len(futures) >= prefetch:
                r0, f0 = futures.popleft()
                pil, size = f0.result()
                yield r0, pil, size
        while futures:
            r0, f0 = futures.popleft()
            pil, size = f0.result()
            yield r0, pil, size

def process_image(
    image_url: str,
    model: YOLO,
    *,
    device: Optional[str] = None,
    # 1ª passada (iguais ao script)
    conf: float = 0.25,
    iou: float = 0.50,
    imgsz: int = 0,     # 0 -> default do modelo
    max_det: int = 5,
    # fallback (indulgente)
    conf_fb: Optional[float] = None,   # None -> max(0.05, conf*0.6)
    iou_fb: Optional[float] = None,    # None -> min(0.70, iou+0.10)
    imgsz_fb: Optional[int] = None,    # None -> max(1280, imgsz or 640)
    # filtros geométricos (iguais ao script)
    min_area_ratio: float = 0.05,
    min_side_px: int = 32,
    max_ar: float = 6.0,
    expand: float = 0.10,
    # classes COCO
    classes: Optional[List[int]] = None,
    box_select_mode: str = "conf_area",
) -> Optional[Dict[str, Any]]:
    """
    Processa uma única imagem (URL) com a mesma lógica do treino.
    Retorna dict com:
      - crop: PIL.Image
      - bbox_xyxy: [x1,y1,x2,y2] (expandida)
      - bbox_raw_xyxy: [x1,y1,x2,y2] (detecção original)
      - conf, cls, cls_name, area_ratio, image_size, params
    Ou None se nada válido após filtros.
    """
    pil = load_pil_from_url(image_url)
    if pil is None:
        return None
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    w, h = pil.size

    # 1ª passada
    res1 = model.predict(
        pil,
        verbose=False,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        classes=classes,
        imgsz=(imgsz or None),
    )
    boxes = res1 and res1[0].boxes
    has_det = bool(boxes) and len(boxes) > 0

    # fallback indulgente
    if not has_det:
        conf_fb = max(0.05, conf * 0.6) if conf_fb is None else conf_fb
        iou_fb  = min(0.70, iou + 0.10) if iou_fb is None else iou_fb
        imgsz_fb = max(1280, (imgsz or 640)) if imgsz_fb is None else imgsz_fb
        res2 = model.predict(
            pil,
            verbose=False,
            device=device,
            conf=conf_fb,
            iou=iou_fb,
            max_det=max(15, max_det),
            classes=classes,
            imgsz=imgsz_fb,
        )
        boxes = res2 and res2[0].boxes
        has_det = bool(boxes) and len(boxes) > 0

    if not has_det:
        return None

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().tolist()
    clss = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []

    # filtros geométricos
    areas, keep_idx = [], []
    for j, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        if bw < min_side_px or bh < min_side_px:
            continue
        big, small = (bw, bh) if bw >= bh else (bh, bw)
        if small == 0 or (big / small) > max_ar:
            continue
        areas.append(bw * bh)
        keep_idx.append(j)
    if not keep_idx:
        return None

    confs_kept = [confs[j] for j in keep_idx]
    areas_kept = [areas[j] for j in range(len(keep_idx))]
    sel_local = choose_box(confs_kept, areas_kept, mode=box_select_mode)
    j_sel = keep_idx[sel_local]
    x1, y1, x2, y2 = boxes_xyxy[j_sel]

    area = max(1.0, (x2 - x1) * (y2 - y1))
    area_ratio = float(area / (w * h + 1e-9))
    if area_ratio < min_area_ratio:
        return None

    ex1, ey1, ex2, ey2 = expand_and_clamp_bbox((x1, y1, x2, y2), w, h, expand)
    crop = pil.crop((ex1, ey1, ex2, ey2))

    cls_name = (COCO80[int(clss[j_sel])] if clss and 0 <= int(clss[j_sel]) < len(COCO80) else None)

    return {
        "crop": crop,
        "bbox_xyxy": [int(ex1), int(ey1), int(ex2), int(ey2)],          # expandida
        "bbox_raw_xyxy": [int(x1), int(y1), int(x2), int(y2)],          # original
        "conf": float(confs[j_sel]),
        "cls": int(clss[j_sel]) if clss else None,
        "cls_name": cls_name,
        "area_ratio": float(area_ratio),
        "image_size": [w, h],
        "params": {
            "conf": conf, "iou": iou, "imgsz": (imgsz or None),
            "conf_fb": conf_fb, "iou_fb": iou_fb, "imgsz_fb": imgsz_fb,
            "max_det": max_det, "expand": expand,
            "min_area_ratio": min_area_ratio, "min_side_px": min_side_px, "max_ar": max_ar,
            "classes": classes, "box_select_mode": box_select_mode,
        }
    }

def main():
    p = argparse.ArgumentParser(description="Preprocessa, filtra e recorta imagens com YOLOv8 (COCO opcional).")
    p.add_argument("--csv_glob", required=True, help='Ex: "data/*_location_fixed.csv"')
    p.add_argument("--out_dir", default="data/processed_dataset_yolo")
    p.add_argument("--yolo_model", default="yolov8m.pt")
    p.add_argument("--device", default=None, help='Ex: "cuda", "mps", "cpu"')
    p.add_argument("--conf", type=float, default=0.25, help="Confiança mínima")
    p.add_argument("--iou", type=float, default=0.50, help="IoU para NMS")
    p.add_argument("--imgsz", type=int, default=0, help="Tamanho de entrada (ex.: 640, 1024, 1280). 0 = default do modelo")
    p.add_argument("--max_det", type=int, default=5, help="Máximo de detecções por imagem")

    p.add_argument("--classes", type=str, default="", help="Nomes ou índices COCO separados por vírgula (ex.: 'cat,dog' ou '15,16')")
    p.add_argument("--use_coco_animals", action="store_true",
                   help=f"Atalho: restringe a classes animais COCO {COCO_ANIMAL_NAMES}")
    p.add_argument("--batch", type=int, default=4, help="Tamanho do lote de inferência (imagens em memória)")
    p.add_argument("--prefetch", type=int, default=32, help="Quantas imagens baixar/abrir em paralelo")
    p.add_argument("--threads", type=int, default=8, help="Threads para prefetch (I/O)")
    p.add_argument("--min_area_ratio", type=float, default=0.05)
    p.add_argument("--min_side_px", type=int, default=32, help="Lado mínimo do crop")
    p.add_argument("--max_ar", type=float, default=6.0, help="Aspect ratio máximo (lado maior / menor)")
    p.add_argument("--expand", type=float, default=0.10, help="Margem relativa da caixa")
    p.add_argument("--box_select_mode", choices=["conf", "area", "conf_area"], default="conf_area")
    p.add_argument("--skip_existing", action="store_true")

    p.add_argument("--save_on_empty", choices=["none","full","center"], default="none",
                   help="Salvar mesmo sem detecção: 'full' (imagem inteira), 'center' (center-crop), ou 'none' (descarta)")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    processed_images_dir = out_dir / "images"
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    # Resolver classes
    classes: Optional[List[int]] = parse_classes_arg(args.classes)
    if args.use_coco_animals:
        animal_idxs = [COCO80.index(n) for n in COCO_ANIMAL_NAMES]
        classes = sorted(set(animal_idxs if classes is None else classes + animal_idxs))

    print(f"[YOLO] Carregando modelo: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    csv_paths = sorted(glob(args.csv_glob))
    if not csv_paths:
        raise SystemExit(f"Nenhum CSV encontrado para: {args.csv_glob}")

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    print(f"Encontradas {len(df)} imagens de {len(csv_paths)} CSVs.")

    processed_records = []
    rejected_small = 0
    rejected_noobj = 0
    rejected_rules = 0
    error_count = 0
    skipped = 0

    rows_iter = df.itertuples()
    pbar = tqdm(total=len(df), desc="Detectando e recortando")
    pref_it = prefetch_rows(rows_iter, prefetch=args.prefetch, max_workers=args.threads)

    for bundle in batched(pref_it, args.batch):
        # bundle é uma lista de tuplas (row, pil, (w,h))
        rows_batch, imgs_batch, sizes_batch = [], [], []
        meta_skip_flags = []

        # Pré-processo: montar o batch e pular existentes
        for (row, pil, size) in bundle:
            if pil is None:
                error_count += 1
                pbar.update(1)
                continue

            w, h = size
            image_filename = f"{sha1_hex(row.image_url)}.jpg"
            save_path = processed_images_dir / image_filename

            if args.skip_existing and save_path.exists():
                skipped += 1
                pr = row._asdict()
                pr["processed_image_path"] = str(save_path)
                pr["yolo_meta"] = json.dumps({"skipped": True})
                processed_records.append(pr)
                pbar.update(1)
                continue

            rows_batch.append((row, save_path, w, h))
            imgs_batch.append(pil)
            sizes_batch.append((w, h))

        if not imgs_batch:
            continue

        # ---------- Inferência em LOTE (1ª passada) ----------
        res_list = model.predict(
            imgs_batch,
            verbose=False,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            classes=classes,
            imgsz=args.imgsz or None,
            batch=args.batch,      # <- usa o batch size
            workers=0              # imgs em memória; dataloader de arquivo não é usado
        )

        # Checar quais falharam e montar fallback batch
        fallback_idxs = []
        for i, res in enumerate(res_list):
            if not res or not res.boxes or len(res.boxes) == 0:
                fallback_idxs.append(i)

        if fallback_idxs:
            imgs_fallback = [imgs_batch[i] for i in fallback_idxs]
            res_fb = model.predict(
                imgs_fallback,
                verbose=False,
                device=args.device,
                conf=max(0.05, args.conf * 0.6),
                iou=min(0.70, args.iou + 0.10),
                max_det=max(15, args.max_det),
                classes=classes,
                imgsz=max(1280, args.imgsz or 640),
                batch=min(len(imgs_fallback), args.batch),
                workers=0,
                # augment=True,  # se disponível
            )
            # substituir nos lugares corretos
            for k, i_orig in enumerate(fallback_idxs):
                res_list[i_orig] = res_fb[k]

        # ---------- Pós-processamento por item ----------
        for i, res in enumerate(res_list):
            row, save_path, w, h = rows_batch[i]
            pil = imgs_batch[i]

            boxes = res and res.boxes
            has_det = bool(boxes) and len(boxes) > 0

            if not has_det:
                if args.save_on_empty != "none":
                    if args.save_on_empty == "center":
                        crop = center_crop(pil, 0.8)
                        crop.save(save_path, quality=95)
                        meta = {"fallback": "center_crop_no_detection"}
                    else:
                        pil.save(save_path, quality=95)
                        meta = {"fallback": "full_image_no_detection"}
                    pr = row._asdict()
                    pr["processed_image_path"] = str(save_path)
                    pr["yolo_meta"] = json.dumps(meta)
                    processed_records.append(pr)
                    pbar.update(1)
                    continue

                rejected_noobj += 1
                pbar.update(1)
                continue

            # === (daqui pra frente, igual ao seu script anterior) ===
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy().tolist()
            clss = boxes.cls.cpu().numpy().tolist() if boxes.cls is not None else []

            areas, keep_idx = [], []
            for j, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                bw, bh = max(0, x2 - x1), max(0, y2 - y1)
                area = bw * bh
                if bw < args.min_side_px or bh < args.min_side_px:
                    continue
                big, small = (bw, bh) if bw >= bh else (bh, bw)
                if small == 0 or (big / small) > args.max_ar:
                    continue
                areas.append(area)
                keep_idx.append(j)

            if not keep_idx:
                rejected_rules += 1
                pbar.update(1)
                continue

            confs_kept = [confs[j] for j in keep_idx]
            areas_kept = [areas[j] for j in range(len(keep_idx))]
            sel_local = choose_box(confs_kept, areas_kept, mode=args.box_select_mode)
            j_sel = keep_idx[sel_local]
            x1, y1, x2, y2 = boxes_xyxy[j_sel]

            area = max(1.0, (x2 - x1) * (y2 - y1))
            area_ratio = float(area / (w * h + 1e-9))
            if area_ratio < args.min_area_ratio:
                rejected_small += 1
                pbar.update(1)
                continue

            ex1, ey1, ex2, ey2 = expand_and_clamp_bbox((x1, y1, x2, y2), w, h, args.expand)
            crop = pil.crop((ex1, ey1, ex2, ey2))
            crop.save(save_path, quality=95)

            pr = row._asdict()
            pr["processed_image_path"] = str(save_path)
            pr["yolo_meta"] = json.dumps({
                "bbox_xyxy": [int(ex1), int(ey1), int(ex2), int(ey2)],
                "bbox_raw_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(confs[j_sel]),
                "cls": int(clss[j_sel]) if clss else None,
                "cls_name": (COCO80[int(clss[j_sel])] if clss and 0 <= int(clss[j_sel]) < len(COCO80) else None),
                "area_ratio": float(area_ratio),
                "image_size": [w, h],
                "model": args.yolo_model,
                "params": {
                    "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz or None,
                    "max_det": args.max_det, "expand": args.expand,
                    "classes": classes
                }
            })
            processed_records.append(pr)
            pbar.update(1)
    df_processed = pd.DataFrame(processed_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_csv_path = out_dir / "processed_metadata.csv"
    df_processed.to_csv(new_csv_path, index=False)

    print("\n--- Pré-processamento Concluído ---")
    print(f"Aceitas e recortadas: {len(df_processed)}")
    print(f"Rejeitadas (sem objeto): {rejected_noobj}")
    print(f"Rejeitadas (regras min_side/ar): {rejected_rules}")
    print(f"Rejeitadas (área < min_area_ratio): {rejected_small}")
    print(f"Ignoradas por erro: {error_count}")
    print(f"Puladas por já existirem: {skipped}")
    print(f"CSV salvo em: {new_csv_path}")
    print(f"Imagens salvas em: {processed_images_dir}")


if __name__ == "__main__":
    main()
