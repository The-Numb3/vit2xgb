# pca_utils.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA

class FoldPCA:
    """
    K-Fold에서 누수 없이 쓰기 위한 얇은 래퍼.
    - fold마다 train으로 fit, train/val 모두 transform
    """
    def __init__(self, n_components: int = 32, whiten: bool = False, random_state: int = 42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self._pca: Optional[PCA] = None

    def fit(self, X: np.ndarray):
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self._pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        return self._pca.fit_transform(X)

def apply_pca_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_components: int = 32,
    whiten: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    한 번에 train/val 변환 (fold 내부에서 사용 편의용)
    """
    p = FoldPCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Xtr = p.fit_transform(X_train)
    Xte = p.transform(X_val)
    return Xtr, Xte
