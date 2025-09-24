#!/usr/bin/env python3
"""
중복 제거된 데이터의 K-fold 교차검증 결과 요약
"""

import pandas as pd
import numpy as np

def main():
    print("=== 중복 제거된 데이터 K-fold 교차검증 결과 요약 ===\n")
    
    # 결과 데이터 (kfold_xgb_eval.py 실행 결과)
    results = {
        'Target': ['Marbling', 'Total', 'Meat Color', 'Texture', 'Surface Moisture'],
        'MAE_mean': [0.6133, 0.6467, 0.6533, 0.6667, 0.5333],
        'MAE_std': [0.0909, 0.0452, 0.0618, 0.0471, 0.0558],
        'R2_mean': [-0.0169, -0.1398, -0.0547, -0.0466, -0.1266],
        'R2_std': [0.1420, 0.1742, 0.2824, 0.0993, 0.0606],
        'ACC_mean': [0.4800, 0.4533, 0.4467, 0.4067, 0.4733],
        'ACC_std': [0.0581, 0.0499, 0.0542, 0.0712, 0.0611],
        'F1_mean': [0.1782, 0.1962, 0.1747, 0.2112, 0.2850],
        'F1_std': [0.0400, 0.0612, 0.0114, 0.0497, 0.0641],
        'QWK_mean': [0.0428, 0.0724, 0.1241, 0.1537, 0.0614],
        'QWK_std': [0.1468, 0.1261, 0.1749, 0.0717, 0.1228]
    }
    
    df = pd.DataFrame(results)
    
    print("📊 성능 지표 요약 (5-fold Cross Validation)")
    print("=" * 80)
    print(f"{'Target':<15} {'MAE':<12} {'R²':<12} {'Accuracy':<12} {'F1':<12} {'QWK':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        mae_str = f"{row['MAE_mean']:.3f}±{row['MAE_std']:.3f}"
        r2_str = f"{row['R2_mean']:.3f}±{row['R2_std']:.3f}"
        acc_str = f"{row['ACC_mean']:.3f}±{row['ACC_std']:.3f}"
        f1_str = f"{row['F1_mean']:.3f}±{row['F1_std']:.3f}"
        qwk_str = f"{row['QWK_mean']:.3f}±{row['QWK_std']:.3f}"
        
        print(f"{row['Target']:<15} {mae_str:<12} {r2_str:<12} {acc_str:<12} {f1_str:<12} {qwk_str:<12}")
    
    # 전체 평가
    print("\n" + "=" * 80)
    print("📈 전체 성능 분석")
    print("=" * 80)
    
    avg_r2 = df['R2_mean'].mean()
    avg_acc = df['ACC_mean'].mean()
    avg_f1 = df['F1_mean'].mean()
    avg_qwk = df['QWK_mean'].mean()
    
    print(f"평균 R²: {avg_r2:.4f}")
    print(f"평균 정확도: {avg_acc:.4f}")
    print(f"평균 F1 점수: {avg_f1:.4f}")
    print(f"평균 QWK: {avg_qwk:.4f}")
    
    # 최고/최저 성능
    best_target = df.loc[df['R2_mean'].idxmax(), 'Target']
    worst_target = df.loc[df['R2_mean'].idxmin(), 'Target']
    
    print(f"\n🏆 최고 성능 타겟: {best_target} (R² = {df['R2_mean'].max():.4f})")
    print(f"📉 최저 성능 타겟: {worst_target} (R² = {df['R2_mean'].min():.4f})")
    
    # 성능 해석
    print("\n" + "=" * 80)
    print("🔍 성능 해석")
    print("=" * 80)
    
    print("주요 관찰 사항:")
    print("1. 모든 타겟에서 R² 값이 음수 또는 0에 가까움 → 예측 성능이 매우 낮음")
    print("2. 정확도는 40-48% 수준 → 랜덤 추측보다 약간 나은 수준")
    print("3. F1 점수가 0.17-0.28로 낮음 → 클래스 불균형 문제")
    print("4. QWK(Quadratic Weighted Kappa)가 낮음 → 순서형 예측 성능 부족")
    
    print("\n가능한 원인 분석:")
    print("• 데이터 크기: 150개 샘플은 1536차원 특성에 비해 매우 작음 (차원의 저주)")
    print("• 특성 품질: ViT 특성이 이 도메인의 품질 평가에 적합하지 않을 수 있음")
    print("• 타겟 분포: 클래스가 불균형하거나 변별력이 부족할 수 있음")
    print("• 오버피팅: 고차원 데이터에서 소규모 데이터로 인한 일반화 부족")
    
    print("\n💡 개선 방안 제안:")
    print("1. 차원 축소: PCA 또는 특성 선택으로 차원 줄이기")
    print("2. 데이터 증강: 더 많은 데이터 수집 또는 증강 기법 적용")
    print("3. 특성 엔지니어링: 도메인 특화 특성 추출")
    print("4. 모델 정규화: L1/L2 정규화, 드롭아웃 등으로 오버피팅 방지")
    print("5. 앙상블 방법: 다양한 모델 조합으로 성능 향상")
    
    # 이전 결과와 비교 (중복 제거 전후)
    print("\n" + "=" * 80)
    print("📋 중복 제거 효과")
    print("=" * 80)
    print("✅ 데이터 품질 개선:")
    print(f"  - 중복 제거: 465개 → 150개 샘플 (315개 중복 제거)")
    print(f"  - 모든 record_key가 고유함 확인")
    print(f"  - 과적합으로 인한 잘못된 고성능 제거")
    
    print("\n🎯 현실적인 성능 측정:")
    print("  - 이전: 인위적으로 높은 점수 (중복 데이터로 인한 착시)")
    print("  - 현재: 실제 모델의 일반화 성능 반영")
    print("  - 모델 개선의 정확한 방향 제시 가능")
    
    # CSV로 저장
    df.to_csv("performance_summary_cleaned.csv", index=False)
    print(f"\n📄 결과가 'performance_summary_cleaned.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()