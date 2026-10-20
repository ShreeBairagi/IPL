"""
First-innings final-score classification module for the IPL Match Intelligence Lab.

Categorizes the final first-innings score into three classes:
- Low (score <= threshold_low)
- Medium (threshold_low < score <= threshold_high)
- High (score > threshold_high)

Thresholds are calculated strictly from training data percentiles (33.33% and 66.67%)
and applied uniformly to held-out test data.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import load_and_preprocess_data
from src.regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    build_regression_pipeline,
    get_regression_models,
    prepare_features_and_target,
    split_first_innings_data,
)
from src.snapshots import create_first_innings_snapshots

CLASS_NAMES: List[str] = ["Low", "Medium", "High"]


def calculate_class_thresholds(
    y_train: pd.Series,
    lower_quantile: float = 1.0 / 3.0,
    upper_quantile: float = 2.0 / 3.0,
) -> Tuple[float, float]:
    """
    Calculate the two score thresholds strictly from the training targets.

    By default, uses exact tertiles (33.33rd and 66.67th percentiles).
    For IPL 2008-2021 training data, this yields exactly 152.0 and 175.0.

    Ensures the test set is NEVER inspected when establishing class boundaries.
    """
    if y_train.empty:
        raise ValueError("y_train cannot be empty when calculating class thresholds.")

    threshold_low = float(y_train.quantile(lower_quantile))
    threshold_high = float(y_train.quantile(upper_quantile))

    if threshold_low >= threshold_high:
        raise ValueError(
            f"Calculated invalid thresholds: lower ({threshold_low}) >= upper ({threshold_high})."
        )

    return threshold_low, threshold_high


def assign_score_classes(
    scores: Union[pd.Series, np.ndarray, List[float]],
    threshold_low: float,
    threshold_high: float,
) -> pd.Series:
    """
    Discretize continuous final scores into three categorical classes:
    - 'Low': score <= threshold_low (boundary inclusive)
    - 'Medium': threshold_low < score <= threshold_high (boundary inclusive)
    - 'High': score > threshold_high

    Documents the explicit boundary convention:
    A score exactly equal to threshold_low (e.g., 152) is assigned to 'Low'.
    A score exactly equal to threshold_high (e.g., 175) is assigned to 'Medium'.
    """
    if threshold_low >= threshold_high:
        raise ValueError("threshold_low must be strictly less than threshold_high.")

    labels: List[str] = []
    for s in scores:
        if s <= threshold_low:
            labels.append("Low")
        elif s <= threshold_high:
            labels.append("Medium")
        else:
            labels.append("High")

    index = scores.index if hasattr(scores, "index") else None
    return pd.Series(labels, index=index, dtype="object")


class MulticlassXGBClassifier(XGBClassifier):
    """
    XGBClassifier extension for multiclass classification with string labels.
    Handles string-to-integer label encoding and decoding safely, keeping
    the class order consistent across training and prediction.
    """
    def __init__(self, classes: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.classes = classes
        self._is_fitting = False

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params.pop("classes", None)
        return params

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "MulticlassXGBClassifier":
        y_arr = np.asarray(y)
        target_classes = self.classes if self.classes is not None else (
            CLASS_NAMES if set(np.unique(y_arr)).issubset(set(CLASS_NAMES)) else np.unique(y_arr).tolist()
        )
        if y_arr.dtype.kind in {"U", "O", "S"}:
            self._classes_arr = np.array(target_classes)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes_arr)}
            self._idx_to_class = {i: c for i, c in enumerate(self._classes_arr)}
            y_enc = np.array([self._class_to_idx[val] for val in y_arr])
        else:
            self._classes_arr = np.unique(y_arr)
            self._class_to_idx = None
            self._idx_to_class = None
            y_enc = y_arr

        self._is_fitting = True
        try:
            super().fit(X, y_enc, **kwargs)
        finally:
            self._is_fitting = False
        return self

    @property
    def classes_(self) -> np.ndarray:
        if hasattr(self, "_classes_arr") and not getattr(self, "_is_fitting", False):
            return self._classes_arr
        return super().classes_

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        preds_enc = super().predict(X, **kwargs)
        if hasattr(self, "_idx_to_class") and self._idx_to_class is not None:
            return np.array([self._idx_to_class[p] for p in preds_enc])
        return preds_enc


def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate standard classifiers for this phase:
    1. Majority Baseline: DummyClassifier predicting the most frequent training class
    2. K-Nearest Neighbors: KNeighborsClassifier (default n_neighbors=5)
    3. Decision Tree: DecisionTreeClassifier (random_state fixed)
    4. Support Vector Machine: SVC (RBF kernel, random_state fixed)
    5. Random Forest: RandomForestClassifier (100 trees, random_state fixed)
    6. XGBoost: MulticlassXGBClassifier (100 estimators, eval_metric="mlogloss", random_state fixed)
    """
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Support Vector Machine": SVC(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": MulticlassXGBClassifier(
            n_estimators=100,
            eval_metric="mlogloss",
            random_state=random_state,
        ),
    }


def build_classification_pipeline(
    classifier: Any,
    preprocessor: Any = None,
    include_toss: bool = False,
) -> Pipeline:
    """
    Wrap the preprocessor and classifier in a standard scikit-learn Pipeline.
    Reuses the ColumnTransformer from regression to standardize numerics and
    one-hot encode categoricals with handle_unknown='ignore'.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor(include_toss=include_toss)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, Any]:
    """
    Calculate classification metrics:
    - Test accuracy
    - Macro-averaged F1-score
    - Confusion matrix ordered as ['Low', 'Medium', 'High']
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "confusion_matrix": cm,
    }


def train_and_evaluate_classifiers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate all classifiers plus Regression-then-cut on train/test data.

    1. Derives thresholds (t1, t2) from train_df targets ONLY.
    2. Discretizes train and test targets into 'Low', 'Medium', 'High'.
    3. Trains and tests the 5 standard classifiers.
    4. Applies Regression-then-cut: reuses the regression estimators (Ridge and Random Forest),
       predicts continuous scores on test set, and discretizes using the same training thresholds.
    """
    # 1. Derive thresholds strictly from training set targets
    t_low, t_high = calculate_class_thresholds(train_df[TARGET_COLUMN])

    # 2. Map targets to discrete classes
    y_train_class = assign_score_classes(train_df[TARGET_COLUMN], t_low, t_high)
    y_test_class = assign_score_classes(test_df[TARGET_COLUMN], t_low, t_high)

    X_train, _ = prepare_features_and_target(train_df, include_toss=include_toss)
    X_test, _ = prepare_features_and_target(test_df, include_toss=include_toss)

    results: Dict[str, Dict[str, Any]] = {}

    # 3. Train and evaluate standard classifiers
    clf_models = get_classification_models(random_state=random_state)
    for name, clf in clf_models.items():
        pipeline = build_classification_pipeline(clf, include_toss=include_toss)
        pipeline.fit(X_train, y_train_class)
        preds = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test_class, preds)
        metrics["pipeline"] = pipeline
        results[name] = metrics

    # 4. Regression-then-cut: reuse existing regression models
    reg_models = get_regression_models(random_state=random_state)
    # Compare both the linear (Ridge) and non-linear (Random Forest) regressors
    for reg_key in ["Ridge Regression", "Random Forest"]:
        reg_pipe = build_regression_pipeline(reg_models[reg_key], include_toss=include_toss)
        reg_pipe.fit(X_train, train_df[TARGET_COLUMN])
        num_preds = reg_pipe.predict(X_test)

        cut_preds = assign_score_classes(num_preds, t_low, t_high)
        cut_metrics = evaluate_predictions(y_test_class, cut_preds)
        cut_metrics["pipeline"] = reg_pipe
        results[f"Regression-then-cut ({reg_key})"] = cut_metrics

    # Class distributions
    train_dist = y_train_class.value_counts()[CLASS_NAMES].to_dict()
    test_dist = y_test_class.value_counts()[CLASS_NAMES].to_dict()

    return {
        "thresholds": {"threshold_low": t_low, "threshold_high": t_high},
        "train_distribution": train_dist,
        "test_distribution": test_dist,
        "models": results,
    }


def run_first_innings_classification(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
    include_toss: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end execution of Phase 2: First-Innings Final-Score Classification.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_first_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_first_innings_data(snapshots_df, split_season=split_season)
    output = train_and_evaluate_classifiers(train_df, test_df, random_state=random_state, include_toss=include_toss)

    output["train_rows"] = len(train_df)
    output["test_rows"] = len(test_df)
    output["train_seasons"] = sorted(train_df["season"].unique().tolist())
    output["test_seasons"] = sorted(test_df["season"].unique().tolist())
    output["include_toss"] = include_toss

    return output


def run_toss_feature_classification_comparison(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
    split_season: int = 2021,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Side-by-side comparison of first-innings final-score classification:
    Baseline (without toss) vs Toss-Enhanced (with toss features).
    Uses identical training and testing matches, identical thresholds, and identical models.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    snapshots_df = create_first_innings_snapshots(matches_df, deliveries_df)

    train_df, test_df = split_first_innings_data(snapshots_df, split_season=split_season)

    baseline_out = train_and_evaluate_classifiers(train_df, test_df, random_state=random_state, include_toss=False)
    toss_out = train_and_evaluate_classifiers(train_df, test_df, random_state=random_state, include_toss=True)

    comparison = {}
    for name in baseline_out["models"].keys():
        b_acc = baseline_out["models"][name]["accuracy"]
        t_acc = toss_out["models"][name]["accuracy"]
        b_f1 = baseline_out["models"][name]["macro_f1"]
        t_f1 = toss_out["models"][name]["macro_f1"]

        comparison[name] = {
            "baseline": {"accuracy": b_acc, "macro_f1": b_f1},
            "with_toss": {"accuracy": t_acc, "macro_f1": t_f1},
            "diff": {
                "accuracy_diff": round(t_acc - b_acc, 4),
                "macro_f1_diff": round(t_f1 - b_f1, 4),
            },
        }

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_matches": sorted(train_df["match_id"].unique().tolist()),
        "test_matches": sorted(test_df["match_id"].unique().tolist()),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "thresholds": baseline_out["thresholds"],
        "comparison": comparison,
    }
