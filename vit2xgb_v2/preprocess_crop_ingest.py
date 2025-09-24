#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_crop_ingest.py

1) result/<이력번호>[_dayN]/<이력번호>.csv 스캔
2) CSV의 TL/TR/BR/BL 좌표로 bbox crop
3) 잘라낸 이미지를 processed/crops/... 에 저장
4) processed/meta.csv 에 증분 추가
5) processed/index.sqlite 로 중복/재처리 방지 (이미지 해시+CSV해시+좌표 조합으로 키 생성)

주의:
- 좌표는 (x,y) 픽셀 기준으로 가정
- 좌표가 바뀌거나 CSV가 바뀌면 record_key가 달라져서 자동 재처리됨
"""

import os, re, sys, glob, csv, json, sqlite3, hashlib, argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import cv2

# ---------- 유틸 ----------

PTN_DAY_IN_DIR = re.compile(r"_day(\d+)$", re.IGNORECASE)

def file_sha1(path: str, block: int = 1024*1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def safe_int_str(x) -> str:
    try: return str(int(float(x)))
    except: return str(x).strip()

def ensure_session_prefix(s: str) -> str:
    s = str(s).strip()
    return s if s.lower().startswith("s") else f"s{s}"

def parse_xy(s) -> Optional[Tuple[int,int]]:
    if pd.isna(s): return None
    if not isinstance(s, str): s = str(s)
    m = re.match(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None

def polygon_to_bbox(pts: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def crop_bbox(img: np.ndarray, bbox: Tuple[int,int,int,int], pad: int = 4) -> np.ndarray:
    H, W = img.shape[:2]
    x1,y1,x2,y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W-1, x2 + pad); y2 = min(H-1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2+1, x1:x2+1, :]

def make_record_key(img_sha1: str, csv_sha1: str, bbox: Tuple[int,int,int,int], extra: Dict[str,Any]) -> str:
    payload = {
        "img_sha1": img_sha1,
        "csv_sha1": csv_sha1,
        "bbox": bbox,
        "extra": extra,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    else:
        # .xlsx
        try:
            return pd.read_excel(path, sheet_name=0, engine="openpyxl")
        except ImportError:
            raise SystemExit("[ERR] openpyxl이 필요합니다. pip install openpyxl")

# ---------- 인덱스/메타 ----------

DDL = """
CREATE TABLE IF NOT EXISTS processed_records(
  record_key TEXT PRIMARY KEY,
  image_path TEXT NOT NULL,
  crop_path  TEXT NOT NULL,
  csv_path   TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

def open_index_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s: conn.execute(s)
    return conn

def record_exists(conn: sqlite3.Connection, record_key: str) -> bool:
    cur = conn.execute("SELECT 1 FROM processed_records WHERE record_key=?", (record_key,))
    return cur.fetchone() is not None

def insert_record(conn: sqlite3.Connection, record_key: str, image_path: str, crop_path: str, csv_path: str):
    conn.execute(
        "INSERT OR IGNORE INTO processed_records(record_key,image_path,crop_path,csv_path,created_at) VALUES(?,?,?,?,?)",
        (record_key, image_path, crop_path, csv_path, datetime.utcnow().isoformat(timespec="seconds"))
    )
    conn.commit()

def append_meta_row(meta_csv: Path, row: Dict[str,Any]):
    # 파일 없으면 헤더와 함께 생성
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = meta_csv.is_file()
    with meta_csv.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# ---------- CSV 파서 ----------

def find_wavelength_cols(df: pd.DataFrame) -> List[int]:
    """CSV의 좌표 컬럼 헤더에서 모든 파장 목록 추출 (예: 'TL (430nm)')"""
    import re
    wl_set = set()
    for c in df.columns:
        s = str(c).strip()
        m = re.match(r"^(TL|TR|BR|BL)\s*\(\s*(\d+)\s*nm\)\s*$", s, flags=re.IGNORECASE)
        if m:
            wl_set.add(int(m.group(2)))
    return sorted(list(wl_set))

def parse_polygon_from_row(row: pd.Series, wl: int) -> Optional[List[Tuple[int,int]]]:
    # 헤더는 대소문자/공백 변형 가능성이 있으므로 매칭
    def find_col(name_pattern: str) -> Optional[str]:
        for c in row.index:
            if str(c).strip().lower() == name_pattern.lower():
                return c
        return None
    keys = [f"TL ({wl}nm)", f"TR ({wl}nm)", f"BR ({wl}nm)", f"BL ({wl}nm)"]
    cols = [find_col(k) for k in keys]
    if any(c is None for c in cols): return None
    pts = [parse_xy(row[c]) for c in cols]
    if any(p is None for p in pts): return None
    return pts

# ---------- 메인 파이프라인 ----------

def process_dataset(dataset_root: Path, out_dir: Path, overwrite: bool=False, pad: int=4):
    result_root = dataset_root / "result"
    if not result_root.is_dir():
        raise SystemExit(f"[ERR] result 폴더가 없습니다: {result_root}")

    meta_csv = out_dir / "meta.csv"
    index_db = out_dir / "index.sqlite"
    crops_dir = out_dir / "crops"
    conn = open_index_db(index_db)

    # 모든 이력번호 폴더에서 CSV 스캔
    csv_list = []
    for p in result_root.glob("*"):
        if not p.is_dir(): continue
        # 폴더 안의 <이력번호>.csv 하나를 찾음
        base = p.name
        # 폴더명이 140093200156_day7 처럼 day 포함일 수 있음 → csv는 <이력번호>.csv로 가정
        slug = base.split("_day")[0]
        cand = p / f"{slug}.xlsx"
        if cand.is_file():
            csv_list.append(cand)

    if not csv_list:
        print("[WARN] 처리할 CSV를 찾지 못했습니다.")
        return

    # 이미지 인덱스 (이름 -> 경로) (프로젝트 전체에서 찾음)
    def build_image_index(root: Path) -> Dict[str,str]:
        idx = {}
        exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
        for ext in exts:
            for ip in root.rglob(ext):
                stem = ip.stem  # 파일명(확장자 제외)
                idx[stem] = str(ip)
        return idx

    img_index = build_image_index(dataset_root)

    total_added, total_skipped = 0, 0

    for i, csv_path in enumerate(csv_list, 1):
        print(f"[{i}/{len(csv_list)}] {csv_path}")
        case_dir = csv_path.parent
        case_name = case_dir.name
        # dayN 추출
        m = PTN_DAY_IN_DIR.search(case_name)
        day = m.group(1) if m else None

        df = load_table(csv_path)
        if "이력번호" not in df.columns or "샘플번호" not in df.columns:
            # 엑셀→CSV 변환 쪽 스키마가 다를 수도 있어 최소 키만 확인
            raise SystemExit(f"[ERR] CSV 필수 컬럼 누락: {csv_path} (이력번호/샘플번호 필요)")

        csv_hash = file_sha1(str(csv_path))
        wls = find_wavelength_cols(df)
        if not wls:
            print(f"[WARN] 좌표 컬럼이 없어 건너뜀: {csv_path}")
            continue

        for _, row in df.iterrows():
            slaughter = safe_int_str(row["이력번호"])
            session = ensure_session_prefix(row["샘플번호"])

            # 경과일수 우선 CSV, 없으면 상위 폴더명에서 추정
            day_row = row.get("경과일수", None)
            if pd.isna(day_row) or str(day_row).strip()=="":
                day_use = day if day is not None else ""
            else:
                day_use = safe_int_str(day_row)

            # 라벨/점수 컬럼(있으면 모두 수집)
            label_cols = [c for c in df.columns if c not in ("이력번호","샘플번호","경과일수")]
            # 이 중 좌표 컬럼은 제외
            coord_like = set()
            for wl in wls:
                for corner in ("TL","TR","BR","BL"):
                    coord_like.add(f"{corner} ({wl}nm)")
            pure_labels = [c for c in label_cols if c not in coord_like]

            for wl in wls:
                pts = parse_polygon_from_row(row, wl)
                if not pts:
                    continue
                bbox = polygon_to_bbox(pts)

                # 원본 이미지 파일명 패턴: <이력번호>_s<세션>_<파장>nm(.png)
                img_stem = f"{slaughter}_{session}_{wl}nm"
                img_path = img_index.get(img_stem, "")
                if not img_path or not os.path.isfile(img_path):
                    # 기존 구조에서 day가 파일명에 들어간 케이스가 있었다면 추가로 시도
                    img_stem2 = f"{slaughter}_{session}_{wl}nm_day{day_use}" if day_use else ""
                    candidate = img_index.get(img_stem2, "")
                    img_path = candidate if (candidate and os.path.isfile(candidate)) else img_path
                if not img_path or not os.path.isfile(img_path):
                    # 마지막으로 case 하위의 이력번호/세션 폴더에서 찾아보기
                    maybe = list((case_dir / slaughter).rglob(f"*{session}*{wl}nm*.png"))
                    if maybe:
                        img_path = str(maybe[0])
                if not img_path or not os.path.isfile(img_path):
                    print(f"[MISS] 이미지 없음: {img_stem} (csv={csv_path})")
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[MISS] 이미지 로드 실패: {img_path}")
                    continue

                crop = crop_bbox(img, bbox, pad=4)

                # 산출 경로: processed/crops/<이력번호[_dayN?]>/<session>/<stem>_crop.png
                case_out = case_name  # day가 붙은 폴더명을 그대로 반영
                out_dir_case = crops_dir / case_out / session
                out_dir_case.mkdir(parents=True, exist_ok=True)
                out_stem = Path(img_path).stem + "_crop"
                out_path = out_dir_case / f"{out_stem}.png"

                if not out_path.is_file() or overwrite:
                    cv2.imwrite(str(out_path), crop)

                # 중복 방지 키 생성
                img_sha1 = file_sha1(img_path)
                record_key = make_record_key(
                    img_sha1=img_sha1,
                    csv_sha1=csv_hash,
                    bbox=bbox,
                    extra={"slaughter": slaughter, "session": session, "day": day_use, "wl": wl}
                )

                if record_exists(conn, record_key) and not overwrite:
                    total_skipped += 1
                    continue

                insert_record(conn, record_key, img_path, str(out_path), str(csv_path))

                # 메타 한 줄 생성 (라벨 포함)
                meta_row = {
                    "record_key": record_key,
                    "slaughter": slaughter,
                    "session": session,
                    "day": day_use,
                    "wavelength_nm": wl,
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "image_path": img_path,
                    "crop_path": str(out_path),
                    "csv_path": str(csv_path),
                    "image_sha1": img_sha1,
                    "csv_sha1": csv_hash,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                }
                # 라벨/스코어 컬럼 추가
                for lc in pure_labels:
                    meta_row[lc] = row.get(lc, "")

                append_meta_row(meta_csv, meta_row)
                total_added += 1

    print(f"[Done] added={total_added}, skipped={total_skipped}")
    print(f"[Meta] {meta_csv}")
    print(f"[Index] {index_db}")
    print(f"[Crops] {crops_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    process_dataset(dataset_root, out_dir, overwrite=bool(args.overwrite))

if __name__ == "__main__":
    main()
