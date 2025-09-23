#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_segcsv.py (fixed)

- sheet_name 미지정 시 첫 번째 시트를 자동 선택 (dict → DataFrame 문제 해결)
- 엑셀에 존재하는 파장(TL/TR/BR/BL (###nm))을 자동 탐지
- image_name: {이력번호}_s{샘플번호}_{wavelength}nm_day{경과일수}
- polygon: "x y; x y; x y; x y" (TL→TR→BR→BL)
"""

import os
import re
import argparse
import pandas as pd

PTN_WL = re.compile(r"^(TL|TR|BR|BL)\s*\(\s*(\d+)\s*nm\)\s*$", re.IGNORECASE)

def parse_xy(s):
    # "(2111,794)" -> (2111, 794)
    if pd.isna(s):
        return None
    if not isinstance(s, str):
        s = str(s)
    m = re.match(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def polygon_str(tl, tr, br, bl):
    def as_xy(t):
        return f"{t[0]} {t[1]}"
    return "; ".join([as_xy(tl), as_xy(tr), as_xy(br), as_xy(bl)])

def find_wavelengths(columns):
    """엑셀 컬럼에서 TL/TR/BR/BL (###nm) 패턴을 찾아 파장 set 반환"""
    wls = set()
    for c in columns:
        m = PTN_WL.match(str(c).strip())
        if m:
            wls.add(int(m.group(2)))
    return sorted(list(wls))

def ensure_session_prefix(session):
    s = str(session).strip()
    return s if s.lower().startswith("s") else f"s{s}"

def as_int_str(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default=None, help="없으면 첫 시트 사용")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label_cols", nargs="*", default=["Marbling","Meat Color","Texture","Surface Moisture","Total"])
    args = ap.parse_args()

    # ✅ sheet 지정 없으면 첫 시트(0)로 강제 → DataFrame 보장
    df = pd.read_excel(args.xlsx, sheet_name=(args.sheet if args.sheet is not None else 0))

    need_base = ["이력번호", "샘플번호", "경과일수"]
    for col in need_base:
        if col not in df.columns:
            raise ValueError(f"엑셀에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # ✅ 엑셀에 실제로 존재하는 파장을 자동 탐지
    wls = find_wavelengths(df.columns)
    if not wls:
        raise RuntimeError("TL/TR/BR/BL (###nm) 형식의 좌표 컬럼을 찾지 못했습니다.")

    out_rows = []
    for _, r in df.iterrows():
        slaughter = as_int_str(r["이력번호"])
        session = ensure_session_prefix(r["샘플번호"])
        day = as_int_str(r["경과일수"])

        # 라벨 수집(있는 것만)
        labels = {}
        for lc in args.label_cols:
            if lc in df.columns:
                labels[lc] = r[lc]

        # 파장별 폴리곤 생성
        for wl in wls:
            keys = [f"TL ({wl}nm)", f"TR ({wl}nm)", f"BR ({wl}nm)", f"BL ({wl}nm)"]
            # 일부 시트는 공백/대소문자 차이가 있을 수 있어 안전하게 찾기
            keymap = {}
            for need in keys:
                found = next((c for c in df.columns if str(c).strip().lower() == need.lower()), None)
                if found is None:
                    keymap = None
                    break
                keymap[need] = found
            if keymap is None:
                continue

            tl = parse_xy(r[keymap[keys[0]]]); tr = parse_xy(r[keymap[keys[1]]])
            br = parse_xy(r[keymap[keys[2]]]); bl = parse_xy(r[keymap[keys[3]]])
            if None in (tl, tr, br, bl):
                # 좌표가 비어있으면 이 파장은 스킵
                continue

            image_name = f"{slaughter}_{session}_{wl}nm_day{day}"
            poly = polygon_str(tl, tr, br, bl)

            row = {
                "image_name": image_name,
                "polygon": poly,
                "wavelength_nm": wl,
                "slaughter": slaughter,
                "session": session,
                "day": day,
            }
            row.update(labels)
            out_rows.append(row)

    if not out_rows:
        raise RuntimeError("내보낼 행이 없습니다. 좌표/필수 컬럼을 확인하세요.")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] Wrote {len(out_df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
