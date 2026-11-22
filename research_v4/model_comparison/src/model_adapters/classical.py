"""
research_v4/model_comparison/src/model_adapters/classical.py

Classical Machine Learning Model Adapters:
1. Ridge, ElasticNet, Linear Regression, Lasso.
2. Support Vector Regression (SVR).
3. Random Forest, ExtraTrees.
4. Gradient Boosting (GBDT), CatBoost, LightGBM, XGBoost.

All feature scaling (StandardScaler) and imputation are fitted strictly on training folds.
Zero test label leakage.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# Gradient boosting frameworks with fallback checks
try:
    import catboost as cb
    _CATBOOST_AVAILABLE = True
except Exception:
    _CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except Exception:
    _LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False


class ClassicalModelAdapter:
    """Base class for classical estimators."""

    def __init__(self, model_name: str, estimator_factory):
        self.model_name = model_name
        self.estimator_factory = estimator_factory
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalModelAdapter":
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = self.estimator_factory()
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test: np.ndarray, max_marks: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        raw_preds = self.model.predict(X_scaled)
        clipped = np.clip(raw_preds, 0.0, max_marks)
        return np.asarray(clipped, dtype=np.float64)


def get_classical_adapters() -> Dict[str, ClassicalModelAdapter]:
    """Factory generating all available classical ML model adapters."""
    adapters = {
        "ridge_regression": ClassicalModelAdapter("Ridge", lambda: Ridge(alpha=10.0, random_state=42)),
        "elastic_net": ClassicalModelAdapter("ElasticNet", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
        "linear_regression": ClassicalModelAdapter("LinearRegression", lambda: LinearRegression()),
        "lasso_regression": ClassicalModelAdapter("Lasso", lambda: Lasso(alpha=0.1, random_state=42)),
        "support_vector_regression": ClassicalModelAdapter("SVR", lambda: SVR(C=2.0, epsilon=0.1, kernel="rbf")),
        "random_forest": ClassicalModelAdapter("RandomForest", lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        "extra_trees": ClassicalModelAdapter("ExtraTrees", lambda: ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42)),
        "gradient_boosting_regressor": ClassicalModelAdapter("GradientBoosting", lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
    }

    if _CATBOOST_AVAILABLE:
        adapters["catboost_regressor"] = ClassicalModelAdapter(
            "CatBoost",
            lambda: cb.CatBoostRegressor(iterations=150, learning_rate=0.05, depth=4, verbose=0, random_seed=42)
        )

    if _LIGHTGBM_AVAILABLE:
        adapters["lightgbm_regressor"] = ClassicalModelAdapter(
            "LightGBM",
            lambda: lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=7, verbose=-1, random_state=42)
        )

    if _XGBOOST_AVAILABLE:
        adapters["xgboost_regressor"] = ClassicalModelAdapter(
            "XGBoost",
            lambda: xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, verbosity=0, random_state=42)
        )

    return adapters
