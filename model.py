"""
Housing price prediction pipeline.
Trains and compares multiple regression models with feature engineering and cross-validation.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class FeatureEngineer:
    """Adds domain-specific engineered features for housing price prediction."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "sqft_living" in df.columns and "sqft_lot" in df.columns:
            df["sqft_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)
        if "bedrooms" in df.columns and "bathrooms" in df.columns:
            df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
        if "yr_built" in df.columns:
            df["age"] = 2024 - df["yr_built"]
        if "yr_renovated" in df.columns:
            df["years_since_renovated"] = 2024 - df["yr_renovated"].replace(0, np.nan)
            df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
        if "sqft_living" in df.columns and "floors" in df.columns:
            df["sqft_per_floor"] = df["sqft_living"] / (df["floors"] + 1)
        return df


class HousingPriceModel:
    """
    Trains and evaluates regression models for housing price prediction.
    Supports log-transform of target, polynomial features, and stacking.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "price", log_transform_target: bool = True):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.log_transform_target = log_transform_target
        self.engineer = FeatureEngineer()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                 self.categorical_features))
        from sklearn.compose import ColumnTransformer
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict[str, Any]:
        models = {
            "Ridge": Ridge(alpha=10.0),
            "Lasso": Lasso(alpha=0.1, max_iter=2000),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                                  max_depth=5, random_state=42,
                                                  tree_method="hist", verbosity=0)
        return models

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        all_num = [c for c in self.numeric_features if c in df.columns]
        all_cat = [c for c in self.categorical_features if c in df.columns]

        X = df[all_num + all_cat]
        y_raw = df[self.target_col]
        y = np.log1p(y_raw) if self.log_transform_target else y_raw

        for col in all_num:
            X[col] = X[col].fillna(X[col].median())
        for col in all_cat:
            X[col] = X[col].fillna("unknown")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        y_test_raw = np.expm1(y_test) if self.log_transform_target else y_test

        preprocessor = self._preprocessor()
        self.results = []

        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            preds_raw = np.expm1(preds) if self.log_transform_target else preds
            preds_raw = np.maximum(preds_raw, 0)

            rmse = float(np.sqrt(mean_squared_error(y_test_raw, preds_raw)))
            mae = float(mean_absolute_error(y_test_raw, preds_raw))
            r2 = float(r2_score(y_test, preds))
            self.models[name] = pipe
            self.results.append({"model": name, "rmse": round(rmse, 0),
                                 "mae": round(mae, 0), "r2": round(r2, 4)})

        results_df = pd.DataFrame(self.results).sort_values("rmse").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model_name not in self.models:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        all_num = [c for c in self.numeric_features if c in df.columns]
        all_cat = [c for c in self.categorical_features if c in df.columns]
        preds = self.models[self.best_model_name].predict(df[all_num + all_cat])
        return np.expm1(preds).astype(int) if self.log_transform_target else preds

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        names = [c for c in self.numeric_features if c in prep.feature_names_in_] + cat_names
        return pd.DataFrame({
            "feature": names[:len(est.feature_importances_)],
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "sqft_living": np.random.uniform(500, 5000, n),
        "sqft_lot": np.random.uniform(1000, 20000, n),
        "bedrooms": np.random.randint(1, 7, n).astype(float),
        "bathrooms": np.random.uniform(1, 4, n).round(1),
        "floors": np.random.choice([1, 1.5, 2, 2.5, 3], n),
        "waterfront": np.random.choice([0, 1], n, p=[0.95, 0.05]).astype(float),
        "yr_built": np.random.randint(1920, 2020, n).astype(float),
        "yr_renovated": np.random.choice([0] * 8 + list(range(1980, 2020)), n).astype(float),
        "condition": np.random.randint(1, 6, n).astype(float),
        "grade": np.random.randint(4, 13, n).astype(float),
        "zipcode": np.random.choice(["98001", "98002", "98003", "98004", "98005"], n),
        "price": np.abs(np.random.lognormal(mean=13.0, sigma=0.5, size=n)),
    })

    model = HousingPriceModel(
        numeric_features=["sqft_living", "sqft_lot", "bedrooms", "bathrooms",
                          "floors", "waterfront", "yr_built", "yr_renovated",
                          "condition", "grade"],
        categorical_features=["zipcode"],
    )
    results = model.fit(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {model.best_model_name}")

    sample_preds = model.predict(df.head(5))
    for p in sample_preds:
        print(f"  Predicted price: ${p:,.0f}")

    imp = model.feature_importance()
    if imp is not None:
        print("\nTop 5 features:")
        print(imp.head(5).to_string(index=False))
