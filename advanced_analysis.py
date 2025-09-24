#!/usr/bin/env python3
"""
과적합 위험도 평가 및 추가 성능 향상 실험
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로딩"""
    features = np.load("vit_out_demo/features.npy")
    meta_df = pd.read_csv("vit_out_demo/meta.csv")
    return features, meta_df

def overfitting_analysis(X, y, target_name):
    """과적합 분석"""
    print(f"\n🔍 과적합 분석: {target_name}")
    print("="*50)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y.astype(int)
    )
    
    # 최적 설정으로 모델 학습
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=1,
        learning_rate=0.0005,
        subsample=0.5,
        colsample_bytree=0.3,
        reg_alpha=2.0,
        reg_lambda=20.0,
        random_state=42,
        verbosity=0
    )
    
    # 학습 곡선 모니터링
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # 성능 지표
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # 과적합 지표
    r2_gap = train_r2 - test_r2
    rmse_gap = test_rmse - train_rmse
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"R² Gap:   {r2_gap:.4f} ({'⚠️ 과적합' if r2_gap > 0.1 else '✅ 정상'})")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")  
    print(f"RMSE Gap:   {rmse_gap:.4f} ({'⚠️ 과적합' if rmse_gap > 0.1 else '✅ 정상'})")
    
    # 과적합 위험도 평가
    risk_level = "낮음"
    if r2_gap > 0.2 or rmse_gap > 0.15:
        risk_level = "높음"
    elif r2_gap > 0.1 or rmse_gap > 0.1:
        risk_level = "중간"
    
    print(f"과적합 위험도: {risk_level}")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2, 
        'r2_gap': r2_gap,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'rmse_gap': rmse_gap,
        'risk_level': risk_level,
        'model': model
    }

def hyperparameter_fine_tuning(X, y, target_name):
    """하이퍼파라미터 미세 조정"""
    print(f"\n🎛️ 하이퍼파라미터 미세 조정: {target_name}")
    print("="*50)
    
    # 테스트할 파라미터 범위
    param_ranges = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'reg_lambda': [5.0, 10.0, 20.0, 50.0],
        'colsample_bytree': [0.2, 0.3, 0.4, 0.5],
    }
    
    best_score = -np.inf
    best_params = None
    results = []
    
    # 기본 파라미터
    base_params = {
        'n_estimators': 1000,
        'max_depth': 1,
        'subsample': 0.5,
        'reg_alpha': 2.0,
        'random_state': 42,
        'verbosity': 0
    }
    
    for lr in param_ranges['learning_rate']:
        for reg_lambda in param_ranges['reg_lambda']:
            for colsample in param_ranges['colsample_bytree']:
                
                params = base_params.copy()
                params.update({
                    'learning_rate': lr,
                    'reg_lambda': reg_lambda,
                    'colsample_bytree': colsample
                })
                
                # 3-fold CV로 빠른 평가
                from sklearn.model_selection import cross_val_score
                model = xgb.XGBRegressor(**params)
                
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                    mean_score = scores.mean()
                    std_score = scores.std()
                    
                    results.append({
                        'lr': lr,
                        'reg_lambda': reg_lambda, 
                        'colsample': colsample,
                        'r2_mean': mean_score,
                        'r2_std': std_score
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params.copy()
                        
                    print(f"LR:{lr:7.4f} | λ:{reg_lambda:4.1f} | Col:{colsample:.1f} | R²:{mean_score:+.4f}±{std_score:.4f}")
                    
                except Exception as e:
                    print(f"LR:{lr:7.4f} | λ:{reg_lambda:4.1f} | Col:{colsample:.1f} | Error: {e}")
    
    print(f"\n🏆 최적 파라미터 (R² = {best_score:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, pd.DataFrame(results)

def ensemble_experiment(X, y, target_name):
    """앙상블 실험"""
    print(f"\n🤖 앙상블 실험: {target_name}")
    print("="*50)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y.astype(int)
    )
    
    # 다양한 설정의 모델들
    models = {
        'Ultra_Conservative': xgb.XGBRegressor(
            n_estimators=2000, max_depth=1, learning_rate=0.0005,
            subsample=0.5, colsample_bytree=0.3, reg_alpha=2.0, reg_lambda=20.0,
            random_state=42, verbosity=0
        ),
        'Medium_Conservative': xgb.XGBRegressor(
            n_estimators=1000, max_depth=2, learning_rate=0.001,
            subsample=0.6, colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=10.0,
            random_state=43, verbosity=0
        ),
        'Mild_Conservative': xgb.XGBRegressor(
            n_estimators=500, max_depth=2, learning_rate=0.005,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=5.0,
            random_state=44, verbosity=0
        )
    }
    
    # 개별 모델 성능
    predictions = {}
    individual_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        individual_scores[name] = r2_score(y_test, pred)
        print(f"{name}: R² = {individual_scores[name]:.4f}")
    
    # 앙상블 (평균)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n🏆 앙상블 (평균): R² = {ensemble_r2:.4f}")
    
    # 가중 앙상블 (성능 기반 가중치)
    weights = np.array(list(individual_scores.values()))
    weights = np.maximum(weights, 0)  # 음수 점수는 0으로
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    
    weighted_pred = np.average(list(predictions.values()), axis=0, weights=weights)
    weighted_r2 = r2_score(y_test, weighted_pred)
    
    print(f"🎯 가중 앙상블: R² = {weighted_r2:.4f}")
    print(f"   가중치: {dict(zip(models.keys(), weights.round(3)))}")
    
    return {
        'individual': individual_scores,
        'ensemble': ensemble_r2,
        'weighted_ensemble': weighted_r2,
        'best_individual': max(individual_scores.values()),
        'ensemble_improvement': ensemble_r2 - max(individual_scores.values())
    }

def main():
    print("🔬 고급 성능 향상 및 과적합 분석")
    print("="*60)
    
    # 데이터 로드
    features, meta_df = load_data()
    
    # Total 타겟 분석
    total_mask = pd.notna(meta_df['Total'])
    X_total = features[total_mask]
    y_total = meta_df.loc[total_mask, 'Total'].values
    
    print(f"분석 데이터: {len(X_total)}개 샘플, {X_total.shape[1]}차원")
    
    # 1. 과적합 분석
    overfitting_results = overfitting_analysis(X_total, y_total, "Total")
    
    # 2. 하이퍼파라미터 미세 조정
    best_params, tuning_results = hyperparameter_fine_tuning(X_total, y_total, "Total")
    
    # 3. 앙상블 실험  
    ensemble_results = ensemble_experiment(X_total, y_total, "Total")
    
    # 결과 요약
    print(f"\n📊 종합 결과 요약")
    print("="*60)
    print(f"과적합 위험도: {overfitting_results['risk_level']}")
    print(f"Train-Test R² Gap: {overfitting_results['r2_gap']:.4f}")
    print(f"최적 단일 모델 R²: {ensemble_results['best_individual']:.4f}")
    print(f"앙상블 R²: {ensemble_results['ensemble']:.4f}")
    print(f"가중 앙상블 R²: {ensemble_results['weighted_ensemble']:.4f}")
    print(f"앙상블 개선: {ensemble_results['ensemble_improvement']:+.4f}")
    
    # 결과 저장
    tuning_results.to_csv("hyperparameter_tuning_results.csv", index=False)
    
    return {
        'overfitting': overfitting_results,
        'tuning': tuning_results,
        'ensemble': ensemble_results
    }

if __name__ == "__main__":
    main()