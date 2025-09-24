#!/usr/bin/env python3
"""
최종 성능 개선 결과 요약
"""

import pandas as pd
import numpy as np

def main():
    print("🏆 육류 품질 예측 모델 성능 개선 - 최종 결과")
    print("=" * 60)
    
    # Baseline vs 최적 조합 비교
    baseline_results = {
        'Target': ['Total', 'Marbling', 'Surface Moisture'],
        'MAE_baseline': [0.647, 0.613, 0.533],
        'R2_baseline': [-0.140, -0.017, -0.127],
        'ACC_baseline': [0.453, 0.480, 0.473],
        'F1_baseline': [0.196, 0.178, 0.285],
        'QWK_baseline': [0.072, 0.043, 0.061],
        # 최적 조합 결과
        'MAE_optimized': [0.640, 0.567, 0.480],
        'R2_optimized': [-0.072, -0.101, -0.015],
        'ACC_optimized': [0.440, 0.520, 0.527],
        'F1_optimized': [0.184, 0.189, 0.331],
        'QWK_optimized': [0.105, 0.053, 0.153],
    }
    
    df = pd.DataFrame(baseline_results)
    
    # 개선율 계산
    df['MAE_improvement'] = ((df['MAE_baseline'] - df['MAE_optimized']) / df['MAE_baseline'] * 100).round(1)
    df['R2_improvement'] = ((df['R2_optimized'] - df['R2_baseline']) / abs(df['R2_baseline']) * 100).round(1)
    df['ACC_improvement'] = ((df['ACC_optimized'] - df['ACC_baseline']) / df['ACC_baseline'] * 100).round(1)
    df['F1_improvement'] = ((df['F1_optimized'] - df['F1_baseline']) / df['F1_baseline'] * 100).round(1)
    df['QWK_improvement'] = ((df['QWK_optimized'] - df['QWK_baseline']) / abs(df['QWK_baseline']) * 100).round(1)
    
    print("\n📊 타겟별 성능 개선 결과")
    print("-" * 60)
    
    for _, row in df.iterrows():
        target = row['Target']
        print(f"\n🎯 {target}")
        print(f"  MAE: {row['MAE_baseline']:.3f} → {row['MAE_optimized']:.3f} ({row['MAE_improvement']:+.1f}%)")
        print(f"  R²:  {row['R2_baseline']:.3f} → {row['R2_optimized']:.3f} ({row['R2_improvement']:+.1f}%)")
        print(f"  ACC: {row['ACC_baseline']:.3f} → {row['ACC_optimized']:.3f} ({row['ACC_improvement']:+.1f}%)")
        print(f"  F1:  {row['F1_baseline']:.3f} → {row['F1_optimized']:.3f} ({row['F1_improvement']:+.1f}%)")
        print(f"  QWK: {row['QWK_baseline']:.3f} → {row['QWK_optimized']:.3f} ({row['QWK_improvement']:+.1f}%)")
    
    # 전체 평균 개선
    print(f"\n🌟 전체 평균 개선율")
    print("-" * 60)
    print(f"MAE 개선:     {df['MAE_improvement'].mean():+.1f}%")
    print(f"R² 개선:      {df['R2_improvement'].mean():+.1f}%") 
    print(f"정확도 개선:   {df['ACC_improvement'].mean():+.1f}%")
    print(f"F1 개선:      {df['F1_improvement'].mean():+.1f}%")
    print(f"QWK 개선:     {df['QWK_improvement'].mean():+.1f}%")
    
    # 최고 성능
    best_r2_idx = df['R2_optimized'].idxmax()
    best_acc_idx = df['ACC_optimized'].idxmax()
    
    print(f"\n🏆 최고 성능")
    print("-" * 60)
    print(f"최고 R²:     {df.iloc[best_r2_idx]['Target']} (R² = {df.iloc[best_r2_idx]['R2_optimized']:.3f})")
    print(f"최고 정확도:  {df.iloc[best_acc_idx]['Target']} (ACC = {df.iloc[best_acc_idx]['ACC_optimized']:.3f})")
    
    # 핵심 발견사항
    print(f"\n💡 핵심 발견사항")
    print("-" * 60)
    print("✅ PCA 32차원 + XGBoost 정규화 조합이 최적")
    print("✅ Surface Moisture가 가장 예측 가능한 타겟")
    print("✅ QWK(순서형 예측 품질) 지표에서 가장 큰 개선")
    print("✅ 모델 안정성 향상 (표준편차 감소)")
    print("⚠️  여전히 절대적 성능은 낮은 수준")
    print("⚠️  그룹별(도축장별) 편차가 매우 큼")
    
    # 향후 계획
    print(f"\n🚀 향후 개선 방향")
    print("-" * 60)
    print("1. 데이터 수집: 최소 500개 이상 샘플 확보")
    print("2. 멀티스펙트럴: 580nm 데이터 활용 확대") 
    print("3. 특성 엔지니어링: 도메인 특화 특성 개발")
    print("4. 앙상블: 다양한 모델 조합으로 성능 향상")
    print("5. 도메인 적응: 도축장별 특성 차이 해결")
    
    # 결과 저장
    df.to_csv("final_performance_comparison.csv", index=False)
    print(f"\n📁 상세 결과가 'final_performance_comparison.csv'에 저장되었습니다.")
    
    print(f"\n{'='*60}")
    print("🎯 성능 개선 프로젝트 완료")
    print("📋 보고서: performance_report.md 참조")
    print("{'='*60}")

if __name__ == "__main__":
    main()