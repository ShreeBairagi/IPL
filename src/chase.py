"""
Chase-win probability prediction at 10 overs for the IPL Match Intelligence Lab.

Predicts whether the team batting second wins the match based exclusively
on the match state at the 10-over snapshot (overs 0-9) of the chase and pre-match context.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.preprocessing import load_and_preprocess_data
from src.snapshots import create_second_innings_snapshots

# Input features available at exactly 10 completed overs of the chase
CHASE_NUMERIC_FEATURES: List[str] = [
    "target",
    "current_score",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
    "runs_needed",
    "required_run_rate",
    "current_run_rate",
]

CHASE_CATEGORICAL_FEATURES: List[str] = [
    "batting_team",
    "bowling_team",
    "venue",
]

# Pre-match toss features for optional evaluation
CHASE_TOSS_CATEGORICAL_FEATURES: List[str] = [
    "toss_winner",
    "toss_decision",
]

# Explicit binary target: 1 = chasing team won, 0 = defending team won
CHASE_TARGET_COLUMN: str = "target_achieved"

# Metadata and outcome columns strictly excluded from the feature matrix X
CHASE_METADATA_COLUMNS: List[str] = [
    "match_id",
    "season",
    "city",
    "winner",
    "balls_remaining",  # constant (60) at 10 completed overs of 20-over match
]


def split_second_innings_data(
    snapshots_df: pd.DataFrame,
    split_season: int = 2021,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split second-innings snapshots chronologically.

    - Training set: seasons <= split_season (e.g., 2008 to 2021)
    - Test set: seasons > split_season (e.g., 2022 to 2024)

    Preserves chronological partition integrity and verifies zero match overlap.
    """
    train_mask = snapshots_df["season"] <= split_season
    test_mask = snapshots_df["season"] > split_season

    train_df = snapshots_df[train_mask].copy()
    test_df = snapshots_df[test_mask].copy()

    # Integrity check: zero match overlap between train and test
    train_matches = set(train_df["match_id"])
    test_matches = set(test_df["match_id"])
    overlap = train_matches.intersection(test_matches)
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping match IDs between train and test splits: {overlap}")

    return train_df, test_df


def prepare_chase_features_and_target(
    df: pd.DataFrame,
    include_toss: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate chase feature matrix X from binary target vector y.

    Leakage Prevention:
    - target_achieved is extracted as target y and strictly excluded from X.
    - winner, result metadata, match_id, season, and city are excluded from X.
    - Only approved 10-over snapshot features are returned.
    - Supports optional inclusion of pre-match toss features.
    """
    feature_cols = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
    if include_toss:
        feature_cols = feature_cols + CHASE_TOSS_CATEGORICAL_FEATURES

    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required chase feature columns: {sorted(missing_cols)}")

    if CHASE_TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column '{CHASE_TARGET_COLUMN}' in dataset.")

    X = df[feature_cols].copy()
    y = df[CHASE_TARGET_COLUMN].copy().astype(int)

    return X, y


def build_chase_preprocessor(include_toss: bool = False) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer for chase prediction:
    - Standardizes numeric features (target, scores, wickets, required run rate, etc.).
    - One-hot encodes categorical features (batting_team, bowling_team, venue)
      with handle_unknown='ignore' so unseen teams and venues in future seasons
      do not raise inference errors.
    - Optionally includes toss features in categorical encoding.
    """
    cat_cols = CHASE_CATEGORICAL_FEATURES + (CHASE_TOSS_CATEGORICAL_FEATURES if include_toss else [])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CHASE_NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def get_chase_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate standard scikit-learn probabilistic classifiers for chase prediction:
    1. Majority Baseline: DummyClassifier predicting most frequent training outcome
    2. Logistic Regression: Standard linear-in-log-odds probabilistic benchmark
    3. K-Nearest Neighbors: Instance-based classifier
    4. Random Forest: Bagged decision trees ensemble with fixed random state
    5. Gradient Boosting: Boosted decision trees ensemble with fixed random state
    6. XGBoost: Boosted decision trees ensemble with logloss eval_metric and fixed seed
    """
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def build_chase_pipeline(
    classifier: Any,
    preprocessor: ColumnTransformer = None,
    include_toss: bool = False,
) -> Pipeline:
    """
    Combine preprocessor and classifier into a single scikit-learn Pipeline.
    Ensures preprocessor is fit strictly on training fold data without leakage.
    """
    if preprocessor is None:
        preprocessor = build_chase_preprocessor(include_toss=include_toss)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_chase_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate classification and probabilistic quality:
    - Accuracy: Overall proportion of correct predictions
    - Precision: TP / (TP + FP) for chase win
    - Recall: TP / (TP + FN) for chase win
    - F1-Score: Harmonic mean of precision and recall
    - Confusion Matrix: [[TN, FP], [FN, TP]] for classes [0, 1]
    - ROC-AUC: Area under ROC curve measuring ranking ability across thresholds
    - Brier Score: Mean squared error of predicted probabilities: sum((p_i - y_i)^2) / N
      Measures probabilistic accuracy (lower is better; 0 = perfect, 0.25 = random coin toss)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
    }


def train_and_evaluate_chase_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate all chase-win prediction models.

    - Fits pipelines strictly on training data (seasons 2008-2021).
    - Evaluates on held-out test data (seasons 2022-2024).
    - Returns classification metrics, probabilistic Brier scores, and fitted pipelines.
    """
    X_train, y_train = prepare_chase_features_and_target(train_df, include_toss=include_toss)
    X_test, y_test = prepare_chase_features_and_target(test_df, include_toss=include_toss)

    models = get_chase_models(random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}

    for name, clf in models.items():
        pipeline = build_chase_pipeline(clf, include_toss=include_toss)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        # Probabilities for positive class (1: chase won)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
        else:
            probs = preds.astype(float)

        metrics = evaluate_chase_predictions(y_test, preds, probs)
        metrics["pipeline"] = pipeline
        results[name] = metrics

    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()

    return {
        "train_distribution": {0: train_dist.get(0, 0), 1: train_dist.get(1, 0)},
        "test_distribution": {0: test_dist.get(0, 0), 1: test_dist.get(1, 0)},
        "models": results,
    }


def run_chase_win_prediction(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end execution of Phase 3: Chase-Win Probability Prediction at 10 Overs.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_second_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_second_innings_data(snapshots_df, split_season=split_season)
    output = train_and_evaluate_chase_models(train_df, test_df, random_state=random_state, include_toss=include_toss)

    output["train_rows"] = len(train_df)
    output["test_rows"] = len(test_df)
    output["train_seasons"] = sorted(train_df["season"].unique().tolist())
    output["test_seasons"] = sorted(test_df["season"].unique().tolist())
    output["include_toss"] = include_toss

    return output


def run_toss_feature_chase_comparison(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Side-by-side comparison of chase-win prediction at 10 overs:
    Baseline (without toss) vs Toss-Enhanced (with toss features).
    Uses identical training and testing matches, identical target, and identical models.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_second_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_second_innings_data(snapshots_df, split_season=split_season)

    baseline_res = train_and_evaluate_chase_models(train_df, test_df, random_state=random_state, include_toss=False)
    toss_res = train_and_evaluate_chase_models(train_df, test_df, random_state=random_state, include_toss=True)

    comparison = {}
    for name in baseline_res["models"].keys():
        b_acc = baseline_res["models"][name]["accuracy"]
        t_acc = toss_res["models"][name]["accuracy"]
        b_auc = baseline_res["models"][name]["roc_auc"]
        t_auc = toss_res["models"][name]["roc_auc"]
        b_f1 = baseline_res["models"][name]["f1"]
        t_f1 = toss_res["models"][name]["f1"]
        b_brier = baseline_res["models"][name]["brier_score"]
        t_brier = toss_res["models"][name]["brier_score"]

        comparison[name] = {
            "baseline": {"accuracy": b_acc, "roc_auc": b_auc, "f1": b_f1, "brier_score": b_brier},
            "with_toss": {"accuracy": t_acc, "roc_auc": t_auc, "f1": t_f1, "brier_score": t_brier},
            "diff": {
                "accuracy_diff": round(t_acc - b_acc, 4),
                "roc_auc_diff": round(t_auc - b_auc, 4),
                "f1_diff": round(t_f1 - b_f1, 4),
                "brier_score_diff": round(t_brier - b_brier, 4),
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
