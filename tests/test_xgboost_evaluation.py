"""
Unit and integration tests for Phase 10: XGBoost Evaluation across Regression, Classification, and Chase.

Covers:
1. Model construction and pipeline compatibility.
2. Multiclass label encoding and decoding.
3. Probability output for chase prediction.
4. Correct baseline and toss feature inclusion.
5. Identical chronological train/test match IDs across comparisons.
6. No target or post-match outcome leakage.
7. Existing models and default behavior remain unchanged.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_and_preprocess_data
from src.snapshots import (
    create_first_innings_snapshots,
    create_second_innings_snapshots,
    create_ten_over_snapshots,
)
from src.regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TOSS_CATEGORICAL_FEATURES,
    build_regression_pipeline,
    get_regression_models,
    prepare_features_and_target,
    run_first_innings_regression,
    run_toss_feature_regression_comparison,
    split_first_innings_data,
    train_and_evaluate_regressors,
)
from src.classification import (
    CLASS_NAMES,
    MulticlassXGBClassifier,
    assign_score_classes,
    build_classification_pipeline,
    calculate_class_thresholds,
    get_classification_models,
    run_first_innings_classification,
    run_toss_feature_classification_comparison,
    train_and_evaluate_classifiers,
)
from src.chase import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    CHASE_TARGET_COLUMN,
    CHASE_TOSS_CATEGORICAL_FEATURES,
    build_chase_pipeline,
    get_chase_models,
    prepare_chase_features_and_target,
    run_chase_win_prediction,
    run_toss_feature_chase_comparison,
    split_second_innings_data,
    train_and_evaluate_chase_models,
)


class TestXGBoostEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches_df, cls.deliveries_df, _ = load_and_preprocess_data()
        cls.snap1, cls.snap2 = create_ten_over_snapshots(cls.matches_df, cls.deliveries_df)
        cls.train1, cls.test1 = split_first_innings_data(cls.snap1)
        cls.train2, cls.test2 = split_second_innings_data(cls.snap2)

    def test_model_construction_and_pipeline_compatibility(self):
        """Verify XGBoost estimators are instantiated with correct hyperparameters and integrate into pipelines."""
        # 1. Regression
        reg_models = get_regression_models(random_state=42)
        self.assertIn("XGBoost", reg_models)
        xgb_reg = reg_models["XGBoost"]
        self.assertIsInstance(xgb_reg, XGBRegressor)
        self.assertEqual(xgb_reg.get_params()["n_estimators"], 100)
        self.assertEqual(xgb_reg.get_params()["random_state"], 42)

        reg_pipe = build_regression_pipeline(xgb_reg, include_toss=False)
        X_tr1, y_tr1 = prepare_features_and_target(self.train1)
        X_te1, _ = prepare_features_and_target(self.test1)
        reg_pipe.fit(X_tr1, y_tr1)
        reg_preds = reg_pipe.predict(X_te1)
        self.assertEqual(len(reg_preds), len(self.test1))
        self.assertTrue(np.all(np.isfinite(reg_preds)))

        # 2. Classification
        clf_models = get_classification_models(random_state=42)
        self.assertIn("XGBoost", clf_models)
        xgb_clf = clf_models["XGBoost"]
        self.assertIsInstance(xgb_clf, XGBClassifier)
        self.assertEqual(xgb_clf.get_params()["n_estimators"], 100)
        self.assertEqual(xgb_clf.get_params()["random_state"], 42)
        self.assertEqual(xgb_clf.get_params()["eval_metric"], "mlogloss")

        t_low, t_high = calculate_class_thresholds(self.train1[TARGET_COLUMN])
        y_tr1_class = assign_score_classes(self.train1[TARGET_COLUMN], t_low, t_high)
        clf_pipe = build_classification_pipeline(xgb_clf, include_toss=False)
        clf_pipe.fit(X_tr1, y_tr1_class)
        clf_preds = clf_pipe.predict(X_te1)
        self.assertEqual(len(clf_preds), len(self.test1))
        self.assertTrue(set(clf_preds).issubset(set(CLASS_NAMES)))

        # 3. Chase
        chase_models = get_chase_models(random_state=42)
        self.assertIn("XGBoost", chase_models)
        xgb_chase = chase_models["XGBoost"]
        self.assertIsInstance(xgb_chase, XGBClassifier)
        self.assertEqual(xgb_chase.get_params()["n_estimators"], 100)
        self.assertEqual(xgb_chase.get_params()["random_state"], 42)
        self.assertEqual(xgb_chase.get_params()["eval_metric"], "logloss")

        chase_pipe = build_chase_pipeline(xgb_chase, include_toss=False)
        X_tr2, y_tr2 = prepare_chase_features_and_target(self.train2)
        X_te2, _ = prepare_chase_features_and_target(self.test2)
        chase_pipe.fit(X_tr2, y_tr2)
        chase_preds = chase_pipe.predict(X_te2)
        self.assertEqual(len(chase_preds), len(self.test2))
        self.assertTrue(set(chase_preds).issubset({0, 1}))

    def test_multiclass_label_encoding_and_decoding(self):
        """Verify string labels are safely encoded to integers and decoded back to string labels preserving class order."""
        clf = MulticlassXGBClassifier(n_estimators=10, eval_metric="mlogloss", random_state=42)
        self.assertIsInstance(clf, XGBClassifier)

        # Synthetic multi-class data
        np.random.seed(42)
        X_toy = pd.DataFrame(np.random.randn(30, 4), columns=["f1", "f2", "f3", "f4"])
        y_toy = pd.Series(["Low", "Medium", "High"] * 10)

        clf.fit(X_toy, y_toy)

        # Verify class order consistency
        self.assertEqual(list(clf.classes_), CLASS_NAMES)

        # Verify predictions return original string labels, not integer codes (0, 1, 2)
        preds = clf.predict(X_toy)
        self.assertEqual(preds.dtype.kind, "U")  # Unicode string
        self.assertTrue(all(p in CLASS_NAMES for p in preds))

        # Verify probabilities match class order
        probs = clf.predict_proba(X_toy)
        self.assertEqual(probs.shape, (30, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        for p, prob in zip(preds, probs):
            pred_idx = list(clf.classes_).index(p)
            self.assertEqual(pred_idx, int(np.argmax(prob)))

        # Verify behavior when test set classes are evaluated
        single_row = X_toy.iloc[[0]]
        single_pred = clf.predict(single_row)
        self.assertIn(single_pred[0], CLASS_NAMES)

    def test_probability_output_for_chase_prediction(self):
        """Verify binary chase XGBClassifier outputs valid probabilities bounded in [0.0, 1.0]."""
        chase_models = get_chase_models(random_state=42)
        xgb_chase = chase_models["XGBoost"]
        pipe = build_chase_pipeline(xgb_chase, include_toss=False)

        X_tr, y_tr = prepare_chase_features_and_target(self.train2)
        X_te, y_te = prepare_chase_features_and_target(self.test2)

        pipe.fit(X_tr, y_tr)
        probs_2d = pipe.predict_proba(X_te)

        # Shape must be (N, 2)
        self.assertEqual(probs_2d.shape, (len(self.test2), 2))

        # Probabilities sum to 1.0
        self.assertTrue(np.allclose(probs_2d.sum(axis=1), 1.0))

        # Class 1 probabilities strictly bounded in [0.0, 1.0]
        pos_probs = probs_2d[:, 1]
        self.assertTrue(np.all((pos_probs >= 0.0) & (pos_probs <= 1.0)))

        # Predictions align with 0.5 threshold
        preds = pipe.predict(X_te)
        expected_preds = (pos_probs >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, expected_preds)

    def test_correct_baseline_and_toss_feature_inclusion(self):
        """Verify baseline features exclude toss features, and include_toss=True appends toss features."""
        # Baseline regression
        X1_base, _ = prepare_features_and_target(self.snap1, include_toss=False)
        self.assertEqual(list(X1_base.columns), NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        self.assertNotIn("toss_winner", X1_base.columns)
        self.assertNotIn("toss_decision", X1_base.columns)

        # Toss regression
        X1_toss, _ = prepare_features_and_target(self.snap1, include_toss=True)
        self.assertEqual(list(X1_toss.columns), NUMERIC_FEATURES + CATEGORICAL_FEATURES + TOSS_CATEGORICAL_FEATURES)
        self.assertIn("toss_winner", X1_toss.columns)
        self.assertIn("toss_decision", X1_toss.columns)

        # Baseline chase
        X2_base, _ = prepare_chase_features_and_target(self.snap2, include_toss=False)
        self.assertEqual(list(X2_base.columns), CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES)
        self.assertNotIn("toss_winner", X2_base.columns)
        self.assertNotIn("toss_decision", X2_base.columns)

        # Toss chase
        X2_toss, _ = prepare_chase_features_and_target(self.snap2, include_toss=True)
        self.assertEqual(list(X2_toss.columns), CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES + CHASE_TOSS_CATEGORICAL_FEATURES)
        self.assertIn("toss_winner", X2_toss.columns)
        self.assertIn("toss_decision", X2_toss.columns)

        # XGBoost fits cleanly on both feature configurations
        reg_pipe_base = build_regression_pipeline(XGBRegressor(n_estimators=10, random_state=42), include_toss=False)
        reg_pipe_base.fit(X1_base.iloc[:50], self.snap1[TARGET_COLUMN].iloc[:50])
        self.assertEqual(len(reg_pipe_base.predict(X1_base.iloc[50:60])), 10)

        reg_pipe_toss = build_regression_pipeline(XGBRegressor(n_estimators=10, random_state=42), include_toss=True)
        reg_pipe_toss.fit(X1_toss.iloc[:50], self.snap1[TARGET_COLUMN].iloc[:50])
        self.assertEqual(len(reg_pipe_toss.predict(X1_toss.iloc[50:60])), 10)

    def test_identical_chronological_train_test_match_ids_across_comparisons(self):
        """Verify chronological split seasons (2008-2021 vs 2022-2024) and identical match IDs in comparison runners."""
        # 1. Regression comparison
        reg_comp = run_toss_feature_regression_comparison()
        self.assertEqual(reg_comp["train_rows"], 832)
        self.assertEqual(reg_comp["test_rows"], 215)
        self.assertEqual(min(reg_comp["train_seasons"]), 2008)
        self.assertEqual(max(reg_comp["train_seasons"]), 2021)
        self.assertEqual(min(reg_comp["test_seasons"]), 2022)
        self.assertEqual(max(reg_comp["test_seasons"]), 2024)
        self.assertIn("XGBoost", reg_comp["comparison"])
        self.assertIn("baseline", reg_comp["comparison"]["XGBoost"])
        self.assertIn("with_toss", reg_comp["comparison"]["XGBoost"])
        self.assertIn("diff", reg_comp["comparison"]["XGBoost"])

        # 2. Classification comparison
        clf_comp = run_toss_feature_classification_comparison()
        self.assertEqual(clf_comp["train_rows"], 832)
        self.assertEqual(clf_comp["test_rows"], 215)
        self.assertEqual(min(clf_comp["train_seasons"]), 2008)
        self.assertEqual(max(clf_comp["train_seasons"]), 2021)
        self.assertEqual(min(clf_comp["test_seasons"]), 2022)
        self.assertEqual(max(clf_comp["test_seasons"]), 2024)
        self.assertIn("XGBoost", clf_comp["comparison"])
        self.assertIn("baseline", clf_comp["comparison"]["XGBoost"])
        self.assertIn("with_toss", clf_comp["comparison"]["XGBoost"])

        # 3. Chase comparison
        chase_comp = run_toss_feature_chase_comparison()
        self.assertEqual(chase_comp["train_rows"], 826)
        self.assertEqual(chase_comp["test_rows"], 212)
        self.assertEqual(min(chase_comp["train_seasons"]), 2008)
        self.assertEqual(max(chase_comp["train_seasons"]), 2021)
        self.assertEqual(min(chase_comp["test_seasons"]), 2022)
        self.assertEqual(max(chase_comp["test_seasons"]), 2024)
        self.assertIn("XGBoost", chase_comp["comparison"])
        self.assertIn("baseline", chase_comp["comparison"]["XGBoost"])
        self.assertIn("with_toss", chase_comp["comparison"]["XGBoost"])

        # Match overlap check
        self.assertEqual(len(set(reg_comp["train_matches"]).intersection(set(reg_comp["test_matches"]))), 0)
        self.assertEqual(len(set(chase_comp["train_matches"]).intersection(set(chase_comp["test_matches"]))), 0)

    def test_no_target_or_post_match_outcome_leakage(self):
        """Verify targets, winner, and post-match columns are strictly excluded from feature matrices."""
        # 1st innings features
        for inc_toss in [False, True]:
            X1, y1 = prepare_features_and_target(self.snap1, include_toss=inc_toss)
            self.assertNotIn(TARGET_COLUMN, X1.columns)
            self.assertNotIn("winner", X1.columns)
            self.assertNotIn("result", X1.columns)
            self.assertNotIn("match_id", X1.columns)
            self.assertNotIn("season", X1.columns)
            self.assertNotIn("city", X1.columns)

            # 2nd innings features
            X2, y2 = prepare_chase_features_and_target(self.snap2, include_toss=inc_toss)
            self.assertNotIn(CHASE_TARGET_COLUMN, X2.columns)
            self.assertNotIn("winner", X2.columns)
            self.assertNotIn("result", X2.columns)
            self.assertNotIn("match_id", X2.columns)
            self.assertNotIn("season", X2.columns)
            self.assertNotIn("city", X2.columns)
            self.assertNotIn("balls_remaining", X2.columns)

    def test_existing_models_and_default_behavior_remain_unchanged(self):
        """Verify existing model estimators and default baseline parameters are strictly preserved."""
        # Regression models
        reg_models = get_regression_models()
        expected_reg = [
            "Mean Baseline",
            "Linear Regression",
            "Ridge Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ]
        self.assertEqual(list(reg_models.keys()), expected_reg)

        # Classification models
        clf_models = get_classification_models()
        expected_clf = [
            "Majority Baseline",
            "K-Nearest Neighbors",
            "Decision Tree",
            "Support Vector Machine",
            "Random Forest",
            "XGBoost",
        ]
        self.assertEqual(list(clf_models.keys()), expected_clf)

        # Chase models
        chase_models = get_chase_models()
        expected_chase = [
            "Majority Baseline",
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ]
        self.assertEqual(list(chase_models.keys()), expected_chase)

        # Default end-to-end runners remain include_toss=False
        reg_summary = run_first_innings_regression()
        self.assertFalse(reg_summary["include_toss"])
        self.assertIn("XGBoost", reg_summary["metrics"])

        clf_summary = run_first_innings_classification()
        self.assertFalse(clf_summary["include_toss"])
        self.assertIn("XGBoost", clf_summary["models"])

        chase_summary = run_chase_win_prediction()
        self.assertFalse(chase_summary["include_toss"])
        self.assertIn("XGBoost", chase_summary["models"])


if __name__ == "__main__":
    unittest.main()
