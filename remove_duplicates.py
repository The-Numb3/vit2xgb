#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복 데이터 제거 스크립트
vit_out_demo의 features.npy와 meta.csv에서 중복된 record_key를 제거
"""
import numpy as np
import pandas as pd
from pathlib import Path

def remove_duplicates(vit_out_dir: str):
    """중복 데이터 제거"""
    vit_out_path = Path(vit_out_dir)
    
    # 파일 경로
    features_file = vit_out_path / "features.npy"
    meta_file = vit_out_path / "meta.csv"
    
    if not features_file.exists() or not meta_file.exists():
        print(f"필요한 파일이 없습니다: {vit_out_path}")
        return
    
    # 데이터 로드
    print("데이터 로드 중...")
    X = np.load(features_file)
    meta_df = pd.read_csv(meta_file)
    
    print(f"원본 데이터: {len(meta_df)}개 샘플, 특징 크기: {X.shape}")
    print(f"고유 record_key: {meta_df['record_key'].nunique()}개")
    print(f"중복 수: {len(meta_df) - meta_df['record_key'].nunique()}개")
    
    # 중복 제거 (첫 번째 항목만 유지)
    unique_mask = ~meta_df['record_key'].duplicated(keep='first')
    meta_clean = meta_df[unique_mask].reset_index(drop=True)
    X_clean = X[unique_mask]
    
    print(f"\n중복 제거 후:")
    print(f"정제된 데이터: {len(meta_clean)}개 샘플, 특징 크기: {X_clean.shape}")
    print(f"고유 record_key: {meta_clean['record_key'].nunique()}개")
    
    # 백업 생성
    backup_dir = vit_out_path / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    print("\n백업 생성 중...")
    np.save(backup_dir / "features_original.npy", X)
    meta_df.to_csv(backup_dir / "meta_original.csv", index=False, encoding="utf-8-sig")
    
    # 정제된 데이터 저장
    print("정제된 데이터 저장 중...")
    np.save(features_file, X_clean)
    meta_clean.to_csv(meta_file, index=False, encoding="utf-8-sig")
    
    print(f"\n완료! 백업은 {backup_dir}에 저장되었습니다.")
    
    # 통계 정보 출력
    print("\n=== 데이터 분석 ===")
    if 'day' in meta_clean.columns:
        print(f"Day별 분포:")
        print(meta_clean['day'].value_counts().sort_index())
    
    if 'wavelength_nm' in meta_clean.columns:
        print(f"\n파장별 분포:")
        print(meta_clean['wavelength_nm'].value_counts().sort_index())
    
    if 'slaughter' in meta_clean.columns:
        print(f"\n도축번호별 분포:")
        print(meta_clean['slaughter'].value_counts())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("사용법: python remove_duplicates.py <vit_out_demo_경로>")
        sys.exit(1)
    
    remove_duplicates(sys.argv[1])