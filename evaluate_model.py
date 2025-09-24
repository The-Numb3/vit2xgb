#!/usr/bin/env python3
"""
중복 제거된 데이터로 XGBoost 모델 성능 평가
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_data(vit_out_dir="vit_out_demo"):
    """임베딩 데이터와 메타데이터 로드"""
    print(f"Loading data from {vit_out_dir}...")
    
    # 특성과 메타데이터 로드
    features = np.load(f"{vit_out_dir}/features.npy")
    meta_df = pd.read_csv(f"{vit_out_dir}/meta.csv")
    
    print(f"Features shape: {features.shape}")
    print(f"Meta records: {len(meta_df)}")
    
    return features, meta_df

def prepare_targets(meta_df):
    """타겟 변수들 준비"""
    targets = {}
    
    # 수치형 타겟들
    numeric_targets = ['Marbling', 'Meat Color', 'Texture', 'Surface Moisture', 'Total']
    
    for target in numeric_targets:
        if target in meta_df.columns:
            # 결측값 제거
            valid_mask = pd.notna(meta_df[target])
            if valid_mask.sum() > 10:  # 최소 10개 샘플이 있는 경우만
                targets[target] = meta_df[target].values
                print(f"{target}: {valid_mask.sum()} valid samples, range: {meta_df[target].min():.1f}-{meta_df[target].max():.1f}")
            else:
                print(f"{target}: insufficient data ({valid_mask.sum()} samples)")
    
    return targets

def evaluate_xgboost(X, y, target_name, test_size=0.2, random_state=42):
    """XGBoost 모델 학습 및 평가"""
    print(f"\n=== Evaluating {target_name} ===")
    
    # 결측값이 있는 샘플 제거
    valid_mask = pd.notna(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_clean) < 10:
        print(f"Insufficient data for {target_name}: {len(X_clean)} samples")
        return None
    
    print(f"Using {len(X_clean)} samples for {target_name}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    
    # Feature Scaling (선택적)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost 모델 학습
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0
    )
    
    # 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 평가 지표 계산
    results = {
        'target': target_name,
        'n_samples': len(X_clean),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }
    
    # Cross Validation
    if len(X_clean) > 20:  # CV를 위한 최소 샘플 수
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        results['cv_r2_mean'] = cv_scores.mean()
        results['cv_r2_std'] = cv_scores.std()
        
        cv_rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                        scoring='neg_root_mean_squared_error')
        results['cv_rmse_mean'] = -cv_rmse_scores.mean()
        results['cv_rmse_std'] = cv_rmse_scores.std()
    
    # 결과 출력
    print(f"Train R²: {results['train_r2']:.4f}")
    print(f"Test R²:  {results['test_r2']:.4f}")
    print(f"Train RMSE: {results['train_rmse']:.4f}")
    print(f"Test RMSE:  {results['test_rmse']:.4f}")
    print(f"Train MAE: {results['train_mae']:.4f}")
    print(f"Test MAE:  {results['test_mae']:.4f}")
    
    if 'cv_r2_mean' in results:
        print(f"CV R² (5-fold): {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        print(f"CV RMSE (5-fold): {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}")
    
    # Feature Importance (top 10)
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    print(f"Top 10 feature indices: {top_features}")
    
    return results, model, scaler, (X_test_scaled, y_test, y_test_pred)

def plot_predictions(target_name, y_true, y_pred, save_path=None):
    """예측 결과 시각화"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 대각선 (perfect prediction line)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{target_name} - Prediction vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R² 표시
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== 중복 제거된 데이터로 모델 성능 평가 ===\n")
    
    # 데이터 로드
    features, meta_df = load_data()
    
    # 데이터 정보 출력
    print(f"\nDataset Info:")
    print(f"- Total samples: {len(features)}")
    print(f"- Feature dimension: {features.shape[1]}")
    print(f"- Wavelengths: {sorted(meta_df['wavelength_nm'].unique())}")
    print(f"- Unique slaughter IDs: {meta_df['slaughter'].nunique()}")
    
    # 타겟 변수 준비
    targets = prepare_targets(meta_df)
    
    if not targets:
        print("No valid target variables found!")
        return
    
    # 각 타겟에 대해 모델 평가
    results_summary = []
    
    for target_name, target_values in targets.items():
        try:
            results, model, scaler, test_data = evaluate_xgboost(features, target_values, target_name)
            
            if results:
                results_summary.append(results)
                
                # 예측 결과 시각화 (Test 데이터만)
                X_test, y_test, y_test_pred = test_data
                plot_predictions(target_name, y_test, y_test_pred, 
                               save_path=f"prediction_{target_name.replace(' ', '_')}.png")
        
        except Exception as e:
            print(f"Error evaluating {target_name}: {e}")
    
    # 결과 요약
    if results_summary:
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        summary_df = pd.DataFrame(results_summary)
        
        # 주요 지표들 출력
        print(f"{'Target':<15} {'Samples':<8} {'Test R²':<10} {'Test RMSE':<12} {'CV R²':<15}")
        print("-" * 60)
        
        for _, row in summary_df.iterrows():
            cv_r2_str = f"{row['cv_r2_mean']:.3f}±{row['cv_r2_std']:.3f}" if 'cv_r2_mean' in row else "N/A"
            print(f"{row['target']:<15} {row['n_samples']:<8} {row['test_r2']:<10.4f} {row['test_rmse']:<12.4f} {cv_r2_str:<15}")
        
        # 결과를 CSV로 저장
        summary_df.to_csv("model_evaluation_results.csv", index=False)
        print(f"\nResults saved to: model_evaluation_results.csv")
        
        # 전체적인 성능 분석
        print(f"\n=== ANALYSIS ===")
        best_r2_target = summary_df.loc[summary_df['test_r2'].idxmax(), 'target']
        best_r2_score = summary_df['test_r2'].max()
        print(f"Best performing target: {best_r2_target} (R² = {best_r2_score:.4f})")
        
        avg_r2 = summary_df['test_r2'].mean()
        print(f"Average R² across all targets: {avg_r2:.4f}")
        
        # 데이터 품질 평가
        if avg_r2 > 0.5:
            print("✅ Good model performance - features are informative")
        elif avg_r2 > 0.2:
            print("⚠️ Moderate performance - some predictive power")
        else:
            print("❌ Low performance - features may need improvement or more data needed")

if __name__ == "__main__":
    main()