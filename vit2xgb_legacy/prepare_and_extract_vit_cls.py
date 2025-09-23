#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_and_extract_vit_cls.py (patched)
- 특징 선택: --feature_type {cls, mean, cls_mean} (default: cls_mean)
- 전처리: mask(폴리곤 마스킹), crop(폴리곤 bbox 크롭), auto(우선순위: crop -> mask)
- 디버그 시각화: --save_debug_vis N (전처리 결과 일부 저장)
"""

import os, re, sys, glob, cv2, argparse, numpy as np, pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, ViTModel

PTN_WL_COL = re.compile(r"^(TL|TR|BR|BL)\s*\(\s*(\d+)\s*nm\)\s*$", re.IGNORECASE)
PTN_DAY_IN_DIR = re.compile(r"_day(\d+)$", re.IGNORECASE)
PTN_IMG_STEM   = re.compile(r"^(?P<slaughter>\d+)_s(?P<session>\d+)_(?P<wavelength>\d+)nm_day(?P<day>\d+)$")

def as_int_str(x) -> str:
    try: return str(int(float(x)))
    except Exception: return str(x).strip()

def ensure_session_prefix(s) -> str:
    s = str(s).strip()
    return s if s.lower().startswith("s") else f"s{s}"

def parse_xy(s):
    if pd.isna(s): return None
    if not isinstance(s, str): s = str(s)
    m = re.match(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s.strip())
    if not m: return None
    return int(m.group(1)), int(m.group(2))

def polygon_str(tl, tr, br, bl) -> str:
    def xy(t): return f"{t[0]} {t[1]}"
    return "; ".join([xy(tl), xy(tr), xy(br), xy(bl)])

def find_wavelengths(columns) -> List[int]:
    wls = set()
    for c in columns:
        m = PTN_WL_COL.match(str(c).strip())
        if m: wls.add(int(m.group(2)))
    return sorted(list(wls))

def parse_filename(stem: str) -> Dict[str, Any]:
    m = PTN_IMG_STEM.match(stem)
    if not m:
        return {"slaughter": None, "session": None, "wavelength_nm": None, "day": None}
    g = m.groupdict()
    return {"slaughter": g["slaughter"], "session": int(g["session"]),
            "wavelength_nm": int(g["wavelength"]), "day": int(g["day"])}

# ---------- 전처리 ----------
def apply_bbox_crop(img: np.ndarray, bbox: Tuple[int,int,int,int], pad: int = 6) -> np.ndarray:
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, xmin - pad); ymin = max(0, ymin - pad)
    xmax = min(w - 1, xmax + pad); ymax = min(h - 1, ymax + pad)
    if xmax <= xmin or ymax <= ymin:
        return img
    return img[ymin:ymax+1, xmin:xmax+1, :]

def polygon_to_bbox(pts: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def apply_polygon_mask(img: np.ndarray, pts: List[Tuple[int,int]]) -> np.ndarray:
    h, w = img.shape[:2]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8)
    out = img.copy(); out[mask == 0] = 0
    return out

def to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ---------- 엑셀 → seg rows ----------
def excel_to_rows(xlsx_path: str, fallback_day_from_dir: Optional[int]=None,
                  label_cols: List[str]=["등급","Marbling","Meat Color","Texture","Surface Moisture","Total"]) -> List[Dict[str, Any]]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except ImportError:
        raise ImportError("openpyxl이 필요합니다. `pip install openpyxl`")
    need_base = ["이력번호", "샘플번호", "경과일수"]
    for col in need_base:
        if col not in df.columns:
            raise ValueError(f"[{xlsx_path}] '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
    wls = find_wavelengths(df.columns)
    if not wls:
        raise RuntimeError(f"[{xlsx_path}] TL/TR/BR/BL (###nm) 좌표 컬럼을 찾지 못했습니다.")

    rows = []
    for _, r in df.iterrows():
        slaughter = as_int_str(r["이력번호"])
        session   = ensure_session_prefix(r["샘플번호"])
        day_raw   = r["경과일수"]
        if pd.isna(day_raw) or str(day_raw).strip() == "":
            if fallback_day_from_dir is None:
                raise ValueError(f"[{xlsx_path}] 경과일수 비어있고 폴더 추정 실패")
            day = str(fallback_day_from_dir)
        else:
            day = as_int_str(day_raw)

        labels = {}
        for lc in label_cols:
            if lc in df.columns:
                labels[lc] = r[lc]

        for wl in wls:
            keys = [f"TL ({wl}nm)", f"TR ({wl}nm)", f"BR ({wl}nm)", f"BL ({wl}nm)"]
            keymap = {}
            for need in keys:
                found = next((c for c in df.columns if str(c).strip().lower() == need.lower()), None)
                if found is None: keymap=None; break
                keymap[need] = found
            if keymap is None: continue

            tl = parse_xy(r[keymap[keys[0]]]); tr = parse_xy(r[keymap[keys[1]]])
            br = parse_xy(r[keymap[keys[2]]]); bl = parse_xy(r[keymap[keys[3]]])
            if None in (tl, tr, br, bl): continue

            image_name = f"{slaughter}_{session}_{wl}nm_day{day}"
            rows.append({
                "image_name": image_name,
                "polygon": polygon_str(tl, tr, br, bl),
                "wavelength_nm": wl, "slaughter": slaughter, "session": session, "day": day,
                **labels
            })
    return rows

# ---------- 이미지 탐색 ----------
def glob_images(base_dir: str) -> Dict[str, str]:
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
    mapping = {}
    for ext in exts:
        for path in glob.iglob(os.path.join(base_dir, "**", ext), recursive=True):
            mapping[os.path.splitext(os.path.basename(path))[0]] = path
    return mapping

# ---------- ViT 추출 ----------
def run_vit(seg_df: pd.DataFrame, image_map: Dict[str,str], out_root: str,
            model_name: str, batch_size: int, preprocess_mode: str, feature_type: str,
            device: str, save_debug_vis: int = 0):
    os.makedirs(out_root, exist_ok=True)
    if preprocess_mode == "auto":  # crop 우선, 없으면 mask
        pmodes = ["crop", "mask"]
    else:
        pmodes = [preprocess_mode]

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device).eval()

    feats_list, meta_rows = [], []
    dbg_left = save_debug_vis

    batch_imgs, batch_meta_idx = [], []

    def flush():
        nonlocal feats_list, batch_imgs, batch_meta_idx
        if not batch_imgs: return
        with torch.no_grad():
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # tokens: [B, 1+P, D]
            tokens = outputs.last_hidden_state
            cls = tokens[:, 0, :]                # [B, D]
            mean_tok = tokens[:, 1:, :].mean(dim=1)  # [B, D]
            if feature_type == "cls":
                feat = cls
            elif feature_type == "mean":
                feat = mean_tok
            else:  # cls_mean
                feat = torch.cat([cls, mean_tok], dim=1)  # [B, 2D]
            feats_list.append(feat.detach().cpu().numpy())
        batch_imgs.clear(); batch_meta_idx.clear()

    pbar = tqdm(seg_df.itertuples(index=False), total=len(seg_df), desc=f"ViT({feature_type})")
    for row in pbar:
        rd = row._asdict() if hasattr(row, "_asdict") else dict(row)
        stem = rd["image_name"]; img_path = image_map.get(stem)
        if not img_path: pbar.set_postfix_str(f"MISS:{stem}"); continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: pbar.set_postfix_str(f"READFAIL:{stem}"); continue

        used_mode = "none"
        processed = img.copy()

        # 폴리곤 파싱
        polygon = rd.get("polygon")
        pts = []
        if isinstance(polygon, str) and polygon.strip():
            try:
                for pair in polygon.split(";"):
                    if pair.strip():
                        x_str, y_str = pair.strip().split()
                        pts.append((int(float(x_str)), int(float(y_str))))
            except Exception:
                pts = []

        # 전처리 적용
        for mode in pmodes:
            if mode == "crop" and pts:
                bbox = polygon_to_bbox(pts)
                processed = apply_bbox_crop(img, bbox)
                used_mode = "crop"; break
            if mode == "mask" and pts:
                processed = apply_polygon_mask(img, pts)
                used_mode = "mask"; break

        pil_img = to_pil_rgb(processed)

        # 메타 (라벨 포함 그대로 보존)
        meta = {k: rd.get(k) for k in seg_df.columns}
        meta.update({"image_path": img_path, "used_mode": used_mode, "feature_type": feature_type})
        meta_rows.append(meta)

        batch_imgs.append(pil_img); batch_meta_idx.append(len(meta_rows)-1)

        # 디버그 저장
        if dbg_left > 0:
            dbg_dir = os.path.join(out_root, "_debug")
            os.makedirs(dbg_dir, exist_ok=True)
            outp = os.path.join(dbg_dir, f"{stem}__{used_mode}.png")
            cv2.imwrite(outp, processed)
            dbg_left -= 1

        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 768), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    np.save(os.path.join(out_root, "features.npy"), feats)
    meta_df.to_csv(os.path.join(out_root, "meta.csv"), index=False, encoding="utf-8-sig")
    print(f"[Done] features: {feats.shape} -> {os.path.join(out_root, 'features.npy')}")
    print(f"[Done] meta: {len(meta_df)} rows -> {os.path.join(out_root, 'meta.csv')}")

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--preprocess_mode", choices=["auto","crop","mask"], default="auto")
    ap.add_argument("--feature_type", choices=["cls","mean","cls_mean"], default="cls_mean")
    ap.add_argument("--save_debug_vis", type=int, default=0, help="전처리 시각화 N장 저장")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    result_root = os.path.join(args.dataset_root, "result")
    if not os.path.isdir(result_root):
        raise FileNotFoundError(f"'result' 폴더가 없습니다: {result_root}")
    os.makedirs(args.out_root, exist_ok=True)

    # 1) 엑셀 수집
    xlsx_list = [p for p in glob.iglob(os.path.join(result_root, "**", "*.xlsx"), recursive=True)]
    if not xlsx_list:
        raise RuntimeError("엑셀(.xlsx)을 찾지 못했습니다.")

    # 2) 엑셀 → rows (day 폴더명에서 보정)
    all_rows = []
    for xlsx in tqdm(xlsx_list, desc="Excel->rows"):
        parent = os.path.dirname(xlsx)
        base   = os.path.basename(parent)  # ex) 140093200156_day7
        m = PTN_DAY_IN_DIR.search(base)
        fb_day = int(m.group(1)) if m else None
        rows = excel_to_rows(xlsx, fallback_day_from_dir=fb_day)
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("세그 rows가 비었습니다.")

    seg_df = pd.DataFrame(all_rows)

    # 3) 이미지 경로 매핑
    img_map = glob_images(args.dataset_root)
    seg_df["image_path"] = seg_df["image_name"].map(lambda s: img_map.get(s, ""))

    # 4) 통합 CSV 저장(라벨 포함)
    seg_csv_path = os.path.join(args.out_root, "combined_seg.csv")
    seg_df.to_csv(seg_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Done] combined seg csv: {seg_csv_path} (rows={len(seg_df)})")

    # 5) ViT 특징 추출
    run_vit(seg_df, img_map, args.out_root, args.model_name, args.batch_size,
            args.preprocess_mode, args.feature_type, args.device, save_debug_vis=args.save_debug_vis)

if __name__ == "__main__":
    main()
