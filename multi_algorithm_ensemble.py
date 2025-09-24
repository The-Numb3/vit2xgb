"""
다중 알고리즘 앙상블 및 고급 피처 엔지니어링 실험
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로드"""
    features = np.load('vit_out_demo/features.npy')
    meta_df = pd.read_csv('vit_out_demo/meta.csv')
    return features, meta_df

def create_diverse_models():
    """다양한 알고리즘 모델 생성"""
    models = {
        'XGBoost_Ultra': xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=1,
            learning_rate=0.0005,
            reg_lambda=20.0,
            reg_alpha=2.0,
            subsample=0.5,
            colsample_bytree=0.3,
            random_state=42,
            verbosity=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.3,
            random_state=42
        ),
        'SVR_RBF': SVR(
            kernel='rbf',
            C=0.1,
            gamma='scale',
            epsilon=0.01
        ),
        'Ridge': Ridge(
            alpha=10.0,
            random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=42
        )
    }
    return models

def feature_engineering_pipeline(X, y=None, method='pca', n_components=100, transformer=None):
    """고급 피처 엔지니어링"""
    
    if method == 'pca':
        if transformer is None:
            # 새로운 PCA 객체 생성 및 fit
            pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]), random_state=42)
            X_transformed = pca.fit_transform(X)
            return X_transformed, pca
        else:
            # 기존 PCA 객체로 transform
            X_transformed = transformer.transform(X)
            return X_transformed
    
    elif method == 'selectk':
        if transformer is None:
            # 새로운 SelectKBest 객체 생성 및 fit
            selector = SelectKBest(score_func=f_regression, k=min(n_components, X.shape[1]))
            X_transformed = selector.fit_transform(X, y)
            return X_transformed, selector
        else:
            # 기존 selector 객체로 transform
            X_transformed = transformer.transform(X)
            return X_transformed
    
    elif method == 'rfe':
        if transformer is None:
            # 새로운 RFE 객체 생성 및 fit
            base_model = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator=base_model, n_features_to_select=min(n_components, X.shape[1]), step=50)
            X_transformed = selector.fit_transform(X, y)
            return X_transformed, selector
        else:
            # 기존 RFE 객체로 transform
            X_transformed = transformer.transform(X)
            return X_transformed

def evaluate_diverse_ensemble(X, y, cv_folds=5):
    """다양한 알고리즘 앙상블 평가"""
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    models = create_diverse_models()
    
    results = []
    ensemble_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🔄 Fold {fold + 1}/{cv_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 피처 엔지니어링
        X_train_pca, pca = feature_engineering_pipeline(X_train, y_train, 'pca', 100)
        X_val_pca = feature_engineering_pipeline(X_val, method='pca', transformer=pca)
        
        # 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_val_scaled = scaler.transform(X_val_pca)
        
        fold_predictions = {}
        fold_scores = {}
        
        # 각 모델 훈련 및 예측
        for name, model in models.items():
            try:
                if 'SVR' in name or 'Ridge' in name or 'ElasticNet' in name:
                    # 선형 모델들은 정규화된 데이터 사용
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                else:
                    # 트리 기반 모델들은 원본 PCA 데이터 사용
                    model.fit(X_train_pca, y_train)
                    pred = model.predict(X_val_pca)
                
                r2 = r2_score(y_val, pred)
                mae = mean_absolute_error(y_val, pred)
                
                fold_predictions[name] = pred
                fold_scores[name] = {'r2': r2, 'mae': mae}
                
                print(f"  {name:15} | R²: {r2:7.4f} | MAE: {mae:.4f}")
                
            except Exception as e:
                print(f"  {name:15} | ERROR: {str(e)}")
                fold_predictions[name] = np.zeros_like(y_val)
                fold_scores[name] = {'r2': -999, 'mae': 999}
        
        # 앙상블 예측
        valid_predictions = {k: v for k, v in fold_predictions.items() 
                           if fold_scores[k]['r2'] > -10}
        
        if valid_predictions:
            # 가중 평균 (R² 기반)
            weights = {}
            total_weight = 0
            for name, pred in valid_predictions.items():
                weight = max(0, fold_scores[name]['r2'] + 1)  # R²가 음수일 때 보정
                weights[name] = weight
                total_weight += weight
            
            if total_weight > 0:
                # 정규화된 가중치
                for name in weights:
                    weights[name] /= total_weight
                
                # 가중 앙상블 예측
                ensemble_pred = np.zeros_like(y_val, dtype=np.float64)
                for name, pred in valid_predictions.items():
                    ensemble_pred += weights[name] * pred.astype(np.float64)
            else:
                # 단순 평균
                ensemble_pred = np.mean(list(valid_predictions.values()), axis=0)
        else:
            ensemble_pred = np.zeros_like(y_val)
        
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        
        print(f"  {'Ensemble':15} | R²: {ensemble_r2:7.4f} | MAE: {ensemble_mae:.4f}")
        print(f"  가중치: {weights if 'weights' in locals() else 'Equal'}")
        print("-" * 60)
        
        results.append({
            'fold': fold,
            'individual': fold_scores,
            'ensemble': {'r2': ensemble_r2, 'mae': ensemble_mae},
            'weights': weights if 'weights' in locals() else {}
        })
    
    return results

def advanced_feature_comparison(X, y):
    """다양한 피처 엔지니어링 방법 비교"""
    
    print("\n🧪 피처 엔지니어링 방법 비교")
    print("=" * 60)
    
    methods = [
        ('PCA_50', 'pca', 50),
        ('PCA_100', 'pca', 100),
        ('PCA_200', 'pca', 200),
        ('SelectK_50', 'selectk', 50),
        ('SelectK_100', 'selectk', 100),
        ('RFE_50', 'rfe', 50)
    ]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for method_name, method, n_comp in methods:
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # 피처 변환
                X_train_trans, transformer = feature_engineering_pipeline(
                    X_train, y_train, method, n_comp
                )
                X_val_trans = feature_engineering_pipeline(X_val, method=method, transformer=transformer)
                
                # XGBoost로 평가
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=1,
                    learning_rate=0.0005,
                    reg_lambda=20.0,
                    random_state=42,
                    verbosity=0
                )
                
                model.fit(X_train_trans, y_train)
                pred = model.predict(X_val_trans)
                r2 = r2_score(y_val, pred)
                scores.append(r2)
                
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                scores.append(-999)
        
        avg_r2 = np.mean(scores) if scores else -999
        std_r2 = np.std(scores) if len(scores) > 1 else 0
        
        print(f"{method_name:12} | R²: {avg_r2:7.4f}±{std_r2:.4f}")

def main():
    """메인 실행 함수"""
    
    print("🚀 고급 다중 알고리즘 앙상블 실험")
    print("=" * 60)
    
    # 데이터 로드
    features, meta_df = load_data()
    targets = ['Total', 'Surface Moisture']
    
    for target in targets:
        print(f"\n🎯 타겟: {target}")
        print("=" * 60)
        
        y = meta_df[target].values
        
        # 피처 엔지니어링 비교
        advanced_feature_comparison(features, y)
        
        # 다중 알고리즘 앙상블 평가
        results = evaluate_diverse_ensemble(features, y)
        
        # 결과 요약
        print(f"\n📊 {target} 최종 결과 요약")
        print("=" * 40)
        
        # 개별 모델 평균 성능
        model_avg = {}
        for result in results:
            for model_name, scores in result['individual'].items():
                if model_name not in model_avg:
                    model_avg[model_name] = []
                model_avg[model_name].append(scores['r2'])
        
        print("개별 모델 평균 성능:")
        for model_name, r2_list in model_avg.items():
            avg_r2 = np.mean(r2_list)
            std_r2 = np.std(r2_list)
            print(f"  {model_name:15} | R²: {avg_r2:7.4f}±{std_r2:.4f}")
        
        # 앙상블 평균 성능
        ensemble_scores = [result['ensemble']['r2'] for result in results]
        ensemble_avg = np.mean(ensemble_scores)
        ensemble_std = np.std(ensemble_scores)
        
        print(f"\n앙상블 성능:")
        print(f"  {'Ensemble':15} | R²: {ensemble_avg:7.4f}±{ensemble_std:.4f}")
        
        # 최고 성능 비교
        best_individual = max([max(r2_list) for r2_list in model_avg.values() if r2_list])
        best_ensemble = max(ensemble_scores)
        
        print(f"\n최고 성능:")
        print(f"  개별 모델 최고: {best_individual:.4f}")
        print(f"  앙상블 최고:   {best_ensemble:.4f}")
        print(f"  개선 효과:     {best_ensemble - best_individual:+.4f}")

if __name__ == "__main__":
    main()