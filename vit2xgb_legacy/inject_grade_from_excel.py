#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원본 엑셀(result/**.xlsx)을 훑어 image_name과 '등급'을 뽑고,
vit_out/meta.csv에 '등급' 컬럼을 병합합니다. (features.npy 재계산 없음)

image_name 규칙: {이력번호}_s{샘플번호}_{파장}nm_day{경과일수}
파장별 좌표 컬럼 패턴: TL (###nm), TR (###nm), BR (###nm), BL (###nm)
"""

import os, re, glob, argparse
import pandas as pd

PTN_WL = re.compile(r"^(TL|TR|BR|BL)\s*\(\s*(\d+)\s*nm\)\s*$", re.IGNORECASE)

def find_wls(cols):
    w=set()
    for c in cols:
        m=PTN_WL.match(str(c).strip())
        if m: w.add(int(m.group(2)))
    return sorted(w)

def as_int_str(x):
    try: return str(int(float(x)))
    except Exception: return str(x).strip()

def ensure_session_prefix(s):
    s=str(s).strip()
    return s if s.lower().startswith("s") else f"s{s}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=".", help="dataset 폴더 경로(내부에 result/ 존재)")
    ap.add_argument("--vit_out", default="./vit_out", help="vit_out 폴더 경로 (meta.csv 위치)")
    args = ap.parse_args()

    result_root = os.path.join(args.dataset_root, "result")
    meta_path = os.path.join(args.vit_out, "meta.csv")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"[ERR] meta.csv을 찾을 수 없습니다: {meta_path}")

    meta = pd.read_csv(meta_path)
    # 이미 등급이 있으면 스킵
    if "등급" in meta.columns and meta["등급"].notna().any():
        print("[Info] meta.csv already has '등급'. Nothing to do.")
        return

    # 모든 엑셀 훑어서 image_name -> 등급 생성
    rows=[]
    xlsx_list = list(glob.iglob(os.path.join(result_root, "**", "*.xlsx"), recursive=True))
    if not xlsx_list:
        raise SystemExit(f"[ERR] 엑셀을 찾지 못했습니다: {result_root}")

    for xlsx in xlsx_list:
        try:
            df = pd.read_excel(xlsx, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"[Warn] read fail: {xlsx} ({e})"); continue

        need = ["이력번호","샘플번호","경과일수"]
        if any(col not in df.columns for col in need):
            print(f"[Warn] missing base cols in {xlsx} -> skip")
            continue

        if "등급" not in df.columns:
            print(f"[Warn] '등급' not in {xlsx} -> skip")
            continue

        wls = find_wls(df.columns)
        if not wls:
            print(f"[Warn] no wavelength cols in {xlsx} -> skip")
            continue

        for _, r in df.iterrows():
            grade = r["등급"]
            slaughter = as_int_str(r["이력번호"])
            session   = ensure_session_prefix(r["샘플번호"])
            day       = as_int_str(r["경과일수"])
            # 각 파장에 동일 등급 부여
            for wl in wls:
                image_name = f"{slaughter}_{session}_{wl}nm_day{day}"
                rows.append({"image_name": image_name, "등급": grade})

    if not rows:
        raise SystemExit("[ERR] 등급을 추출한 행이 없습니다. 엑셀에 '등급' 컬럼/좌표 컬럼을 확인하세요.")

    grade_df = pd.DataFrame(rows).drop_duplicates(subset=["image_name"])
    merged = meta.merge(grade_df, on="image_name", how="left")

    out_path = os.path.join(args.vit_out, "meta.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Done] meta.csv updated with '등급' ({merged['등급'].notna().sum()} rows filled).")

if __name__ == "__main__":
    main()
