"""
First-innings final-score regression models for the IPL Match Intelligence Lab.

Predicts the final first-innings score of an IPL match based exclusively
on the match state at the 10-over snapshot (overs 0-9) and pre-match context.
"""
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.preprocessing import load_and_preprocess_data
from src.snapshots import create_first_innings_snapshots

# Features available at the 10-over snapshot
NUMERIC_FEATURES: List[str] = [
    "current_score",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
]

CATEGORICAL_FEATURES: List[str] = [
    "batting_team",
    "bowling_team",
    "venue",
]

# Pre-match toss features for optional evaluation
TOSS_CATEGORICAL_FEATURES: List[str] = [
    "toss_winner",
    "toss_decision",
]

# Explicit target column - strictly separated from features to prevent leakage
TARGET_COLUMN: str = "final_score"

# Columns used for splitting or tracking, never passed to the ML estimators as features
METADATA_COLUMNS: List[str] = [
    "match_id",
    "season",
    "city",
]


def split_first_innings_data(
    snapshots_df: pd.DataFrame,
    split_season: int = 2021,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split first-innings snapshots chronologically.

    - Training set: seasons <= split_season (e.g., 2008 to 2021)
    - Test set: seasons > split_season (e.g., 2022 to 2024)

    Preserves chronological ordering to prevent temporal data leakage and ensures
    all observations from the same match remain strictly within a single partition.
    """
    train_mask = snapshots_df["season"] <= split_season
    test_mask = snapshots_df["season"] > split_season

    train_df = snapshots_df[train_mask].copy()
    test_df = snapshots_df[test_mask].copy()

    # Integrity check: verify zero match overlap between train and test partitions
    train_matches = set(train_df["match_id"])
    test_matches = set(test_df["match_id"])
    overlap = train_matches.intersection(test_matches)
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping match IDs between train and test splits: {overlap}")

    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame,
    include_toss: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X from target vector y.

    Guarantees:
    - Target (final_score) is never present in X.
    - Match IDs and seasons are excluded from X so the model cannot memorize IDs.
    - Only approved 10-over snapshot features are returned.
    - Optionally includes pre-match toss features.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if include_toss:
        feature_cols = feature_cols + TOSS_CATEGORICAL_FEATURES

    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {sorted(missing_cols)}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_COLUMN}' in dataset.")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_preprocessor(include_toss: bool = False) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer for preprocessing:
    - Standardizes numeric features with StandardScaler.
    - One-hot encodes categorical features with OneHotEncoder.
      handle_unknown='ignore' ensures unseen teams or venues in future seasons
      (e.g., Gujarat Titans, Lucknow Super Giants in 2022+) do not crash inference.
    - Optionally includes toss features in categorical encoding.
    """
    cat_cols = CATEGORICAL_FEATURES + (TOSS_CATEGORICAL_FEATURES if include_toss else [])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate the approved regression estimators:
    1. Mean Baseline: DummyRegressor predicting mean training score
    2. Linear Regression: Ordinary Least Squares regressor
    3. Ridge Regression: L2-regularized linear regressor (alpha=1.0)
    4. Random Forest Regressor: 100-tree ensemble with fixed seed
    5. Gradient Boosting Regressor: 100-estimator gradient boosted trees with fixed seed
    6. XGBoost: 100-estimator extreme gradient boosted trees with fixed seed
    """
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state),
    }


def build_regression_pipeline(
    regressor: Any,
    preprocessor: ColumnTransformer = None,
    include_toss: bool = False,
) -> Pipeline:
    """
    Wrap preprocessor and regressor into a single scikit-learn Pipeline.
    Ensures preprocessor is fit strictly on training fold data without leakage.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor(include_toss=include_toss)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_and_evaluate_regressors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all approved regression models.

    - Fits pipelines strictly on training data (seasons 2008-2021).
    - Evaluates predictions on held-out test data (seasons 2022-2024).
    - Returns actual MAE, RMSE, R^2, and fitted pipelines.
    """
    X_train, y_train = prepare_features_and_target(train_df, include_toss=include_toss)
    X_test, y_test = prepare_features_and_target(test_df, include_toss=include_toss)

    models = get_regression_models(random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}

    for name, regressor in models.items():
        pipeline = build_regression_pipeline(regressor, include_toss=include_toss)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results[name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "pipeline": pipeline,
        }

    return results


def run_first_innings_regression(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end execution of Phase 1: First-Innings Final-Score Regression.
    1. Loads cleaned matches and deliveries.
    2. Builds 10-over first-innings snapshots.
    3. Performs chronological 2008-2021 train / 2022-2024 test split.
    4. Trains and evaluates all 5 models.
    5. Returns split summary and metric results.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_first_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_first_innings_data(snapshots_df, split_season=split_season)
    results = train_and_evaluate_regressors(train_df, test_df, random_state=random_state, include_toss=include_toss)

    summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "include_toss": include_toss,
        "metrics": {
            name: {"mae": res["mae"], "rmse": res["rmse"], "r2": res["r2"]}
            for name, res in results.items()
        },
        "models": results,
    }

    return summary


def run_toss_feature_regression_comparison(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Side-by-side comparison of first-innings final-score regression:
    Baseline (without toss) vs Toss-Enhanced (with toss features).
    Uses identical training and testing matches and the same model settings.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_first_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_first_innings_data(snapshots_df, split_season=split_season)

    baseline_res = train_and_evaluate_regressors(train_df, test_df, random_state=random_state, include_toss=False)
    toss_res = train_and_evaluate_regressors(train_df, test_df, random_state=random_state, include_toss=True)

    comparison = {}
    for name in baseline_res.keys():
        b_mae = baseline_res[name]["mae"]
        t_mae = toss_res[name]["mae"]
        b_rmse = baseline_res[name]["rmse"]
        t_rmse = toss_res[name]["rmse"]
        b_r2 = baseline_res[name]["r2"]
        t_r2 = toss_res[name]["r2"]

        comparison[name] = {
            "baseline": {"mae": b_mae, "rmse": b_rmse, "r2": b_r2},
            "with_toss": {"mae": t_mae, "rmse": t_rmse, "r2": t_r2},
            "diff": {
                "mae_diff": round(t_mae - b_mae, 2),
                "rmse_diff": round(t_rmse - b_rmse, 2),
                "r2_diff": round(t_r2 - b_r2, 4),
            },
        }

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_matches": sorted(train_df["match_id"].unique().tolist()),
        "test_matches": sorted(test_df["match_id"].unique().tolist()),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "comparison": comparison,
    }
