"""
Unit and integration tests for toss feature evaluation across regression, classification, and chase.
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

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
    prepare_features_and_target,
    run_toss_feature_regression_comparison,
)
from src.classification import run_toss_feature_classification_comparison
from src.chase import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    CHASE_TARGET_COLUMN,
    CHASE_TOSS_CATEGORICAL_FEATURES,
    prepare_chase_features_and_target,
    run_toss_feature_chase_comparison,
)


class TestTossFeatureEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches_df, cls.deliveries_df, _ = load_and_preprocess_data()
        cls.snap1, cls.snap2 = create_ten_over_snapshots(cls.matches_df, cls.deliveries_df)

    def test_snapshots_contain_toss_columns_and_no_missing_values(self):
        """Verify that toss_winner and toss_decision are present and have 0 missing values for eligible matches."""
        for name, df in [("1st Innings Snapshots", self.snap1), ("2nd Innings Snapshots", self.snap2)]:
            self.assertIn("toss_winner", df.columns, f"{name} missing toss_winner")
            self.assertIn("toss_decision", df.columns, f"{name} missing toss_decision")
            self.assertEqual(df["toss_winner"].isna().sum(), 0, f"{name} contains null toss_winner")
            self.assertEqual(df["toss_decision"].isna().sum(), 0, f"{name} contains null toss_decision")
            # Verify valid toss decisions
            self.assertTrue(set(df["toss_decision"].unique()).issubset({"bat", "field"}))

    def test_correct_match_id_association(self):
        """Verify toss values on snapshots match the exact match record by match_id."""
        matches_lookup = self.matches_df.set_index("id")[["toss_winner", "toss_decision"]].to_dict(orient="index")

        for sample_snap in [self.snap1.head(25), self.snap2.head(25)]:
            for _, row in sample_snap.iterrows():
                mid = row["match_id"]
                expected_winner = matches_lookup[mid]["toss_winner"]
                expected_decision = matches_lookup[mid]["toss_decision"]
                self.assertEqual(row["toss_winner"], expected_winner)
                self.assertEqual(row["toss_decision"], expected_decision)

    def test_backwards_compatibility_when_include_toss_false(self):
        """Verify feature extraction preserves default behavior without toss when include_toss=False."""
        X1_default, y1 = prepare_features_and_target(self.snap1)
        X1_explicit, _ = prepare_features_and_target(self.snap1, include_toss=False)
        expected_cols1 = NUMERIC_FEATURES + CATEGORICAL_FEATURES

        self.assertEqual(list(X1_default.columns), expected_cols1)
        self.assertEqual(list(X1_explicit.columns), expected_cols1)
        self.assertNotIn("toss_winner", X1_default.columns)
        self.assertNotIn("toss_decision", X1_default.columns)

        X2_default, y2 = prepare_chase_features_and_target(self.snap2)
        X2_explicit, _ = prepare_chase_features_and_target(self.snap2, include_toss=False)
        expected_cols2 = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES

        self.assertEqual(list(X2_default.columns), expected_cols2)
        self.assertEqual(list(X2_explicit.columns), expected_cols2)
        self.assertNotIn("toss_winner", X2_default.columns)
        self.assertNotIn("toss_decision", X2_default.columns)

    def test_correct_feature_inclusion_when_include_toss_true(self):
        """Verify toss features are appended to feature matrix X when include_toss=True."""
        X1, _ = prepare_features_and_target(self.snap1, include_toss=True)
        expected_cols1 = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TOSS_CATEGORICAL_FEATURES
        self.assertEqual(list(X1.columns), expected_cols1)
        self.assertIn("toss_winner", X1.columns)
        self.assertIn("toss_decision", X1.columns)

        X2, _ = prepare_chase_features_and_target(self.snap2, include_toss=True)
        expected_cols2 = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES + CHASE_TOSS_CATEGORICAL_FEATURES
        self.assertEqual(list(X2.columns), expected_cols2)
        self.assertIn("toss_winner", X2.columns)
        self.assertIn("toss_decision", X2.columns)

    def test_no_target_or_outcome_leakage(self):
        """Verify target labels and post-match outcome columns are strictly excluded from X."""
        X1, y1 = prepare_features_and_target(self.snap1, include_toss=True)
        self.assertNotIn(TARGET_COLUMN, X1.columns)
        self.assertNotIn("winner", X1.columns)
        self.assertNotIn("result", X1.columns)

        X2, y2 = prepare_chase_features_and_target(self.snap2, include_toss=True)
        self.assertNotIn(CHASE_TARGET_COLUMN, X2.columns)
        self.assertNotIn("winner", X2.columns)
        self.assertNotIn("result", X2.columns)

    def test_identical_train_test_match_ids_in_comparison_runners(self):
        """Verify comparison runners use identical train/test match IDs for both variants."""
        # 1. Regression comparison
        reg_comp = run_toss_feature_regression_comparison()
        self.assertEqual(reg_comp["train_rows"], 832)
        self.assertEqual(reg_comp["test_rows"], 215)
        self.assertIn("Ridge Regression", reg_comp["comparison"])
        ridge_diff = reg_comp["comparison"]["Ridge Regression"]["diff"]
        self.assertIn("mae_diff", ridge_diff)
        self.assertIn("r2_diff", ridge_diff)
        self.assertIn("rmse_diff", ridge_diff)

        # 2. Chase comparison
        chase_comp = run_toss_feature_chase_comparison()
        self.assertEqual(chase_comp["train_rows"], 826)
        self.assertEqual(chase_comp["test_rows"], 212)
        self.assertIn("Logistic Regression", chase_comp["comparison"])
        log_diff = chase_comp["comparison"]["Logistic Regression"]["diff"]
        self.assertIn("accuracy_diff", log_diff)
        self.assertIn("roc_auc_diff", log_diff)
        self.assertIn("brier_score_diff", log_diff)

        # 3. Classification comparison
        clf_comp = run_toss_feature_classification_comparison()
        self.assertEqual(clf_comp["train_rows"], 832)
        self.assertEqual(clf_comp["test_rows"], 215)
        self.assertIn("Random Forest", clf_comp["comparison"])
        rf_diff = clf_comp["comparison"]["Random Forest"]["diff"]
        self.assertIn("accuracy_diff", rf_diff)
        self.assertIn("macro_f1_diff", rf_diff)


if __name__ == "__main__":
    unittest.main()
