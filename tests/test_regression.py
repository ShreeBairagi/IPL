"""
Unit and integration tests for first-innings final-score regression.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    build_regression_pipeline,
    get_regression_models,
    prepare_features_and_target,
    run_first_innings_regression,
    split_first_innings_data,
    train_and_evaluate_regressors,
)


class TestRegression(unittest.TestCase):

    def setUp(self):
        """Create synthetic first-innings snapshot fixture for unit tests."""
        rows = []
        # Create 14 training seasons (2008-2021) with 2 matches each = 28 rows
        for year in range(2008, 2022):
            for m in [1, 2]:
                rows.append({
                    "match_id": year * 10 + m,
                    "season": year,
                    "venue": "Wankhede Stadium" if m == 1 else "Eden Gardens",
                    "city": "Mumbai" if m == 1 else "Kolkata",
                    "batting_team": "Mumbai Indians" if m == 1 else "Kolkata Knight Riders",
                    "bowling_team": "Chennai Super Kings" if m == 1 else "Delhi Capitals",
                    "current_score": 80 + m * 5,
                    "wickets": 2 + m,
                    "runs_last_5": 40 + m * 2,
                    "wickets_last_5": 1,
                    "final_score": 160 + m * 10,
                })

        # Create 3 test seasons (2022-2024) with 2 matches each = 6 rows
        for year in range(2022, 2025):
            for m in [1, 2]:
                rows.append({
                    "match_id": year * 10 + m,
                    "season": year,
                    "venue": "Narendra Modi Stadium" if m == 1 else "Ekana Stadium",  # new venues
                    "city": "Ahmedabad" if m == 1 else "Lucknow",
                    "batting_team": "Gujarat Titans" if m == 1 else "Lucknow Super Giants",  # new teams
                    "bowling_team": "Rajasthan Royals",
                    "current_score": 85 + m * 5,
                    "wickets": 2 + m,
                    "runs_last_5": 42 + m * 2,
                    "wickets_last_5": 1,
                    "final_score": 170 + m * 10,
                })

        self.synthetic_df = pd.DataFrame(rows)

    def test_chronological_split(self):
        """Verify that split strictly separates 2008-2021 into train and 2022-2024 into test."""
        train_df, test_df = split_first_innings_data(self.synthetic_df, split_season=2021)

        self.assertEqual(len(train_df), 28)
        self.assertEqual(len(test_df), 6)

        self.assertTrue((train_df["season"] <= 2021).all())
        self.assertTrue((test_df["season"] >= 2022).all())
        self.assertEqual(train_df["season"].max(), 2021)
        self.assertEqual(test_df["season"].min(), 2022)

    def test_no_match_overlap_between_splits(self):
        """Verify that no match ID appears in both training and test partitions."""
        train_df, test_df = split_first_innings_data(self.synthetic_df, split_season=2021)

        train_matches = set(train_df["match_id"])
        test_matches = set(test_df["match_id"])
        overlap = train_matches.intersection(test_matches)

        self.assertEqual(len(overlap), 0)

    def test_feature_target_separation_and_leakage_prevention(self):
        """Verify that target and tracking columns are strictly separated from feature set."""
        X, y = prepare_features_and_target(self.synthetic_df)

        # Target column must NOT be in X
        self.assertNotIn(TARGET_COLUMN, X.columns)
        self.assertEqual(y.name, TARGET_COLUMN)

        # Match ID and season metadata must NOT be in X
        self.assertNotIn("match_id", X.columns)
        self.assertNotIn("season", X.columns)
        self.assertNotIn("city", X.columns)

        # Approved feature set must match expected columns exactly
        expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.assertEqual(list(X.columns), expected_features)

        # Missing target or feature raises error
        incomplete_df = self.synthetic_df.drop(columns=[TARGET_COLUMN])
        with self.assertRaises(KeyError):
            prepare_features_and_target(incomplete_df)

    def test_unseen_categories_handling(self):
        """
        Verify that new teams and venues appearing in future seasons
        (e.g., Gujarat Titans, Ekana Stadium) do not raise errors during inference.
        """
        train_df, test_df = split_first_innings_data(self.synthetic_df, split_season=2021)
        models = get_regression_models()
        rf = models["Random Forest"]

        pipeline = build_regression_pipeline(rf)
        X_train, y_train = prepare_features_and_target(train_df)
        X_test, _ = prepare_features_and_target(test_df)

        # Should fit and predict smoothly without ValueError
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        self.assertEqual(len(preds), len(test_df))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_all_five_models_fit_and_evaluate(self):
        """Verify all 5 requested models are evaluated and return valid metrics."""
        train_df, test_df = split_first_innings_data(self.synthetic_df, split_season=2021)
        results = train_and_evaluate_regressors(train_df, test_df)

        expected_models = [
            "Mean Baseline",
            "Linear Regression",
            "Ridge Regression",
            "Random Forest",
            "Gradient Boosting",
        ]

        for model_name in expected_models:
            self.assertIn(model_name, results)
            metrics = results[model_name]
            self.assertIn("mae", metrics)
            self.assertIn("r2", metrics)
            self.assertIsInstance(metrics["mae"], float)
            self.assertIsInstance(metrics["r2"], float)
            self.assertGreater(metrics["mae"], 0.0)

    def test_real_dataset_first_innings_regression(self):
        """Integration test executing full pipeline on actual IPL dataset."""
        summary = run_first_innings_regression()

        # Check train and test split counts (832 train, 215 test)
        self.assertEqual(summary["train_rows"], 832)
        self.assertEqual(summary["test_rows"], 215)
        self.assertEqual(min(summary["train_seasons"]), 2008)
        self.assertEqual(max(summary["train_seasons"]), 2021)
        self.assertEqual(min(summary["test_seasons"]), 2022)
        self.assertEqual(max(summary["test_seasons"]), 2024)

        metrics = summary["metrics"]

        # Mean baseline should have MAE around 30.14
        self.assertAlmostEqual(metrics["Mean Baseline"]["mae"], 30.14, delta=0.5)

        # Regressors should substantially outperform mean baseline (MAE ~17-18 vs ~30)
        for model in ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]:
            self.assertIn(model, metrics)
            mae = metrics[model]["mae"]
            r2 = metrics[model]["r2"]
            self.assertLess(mae, 20.0)
            self.assertGreater(mae, 15.0)
            self.assertGreater(r2, 0.50)


if __name__ == "__main__":
    unittest.main()
