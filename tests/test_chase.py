"""
Unit and integration tests for chase-win probability prediction at 10 overs.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chase import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    CHASE_TARGET_COLUMN,
    build_chase_pipeline,
    build_chase_preprocessor,
    evaluate_chase_predictions,
    get_chase_models,
    prepare_chase_features_and_target,
    run_chase_win_prediction,
    split_second_innings_data,
    train_and_evaluate_chase_models,
)
from src.snapshots import (
    create_second_innings_snapshots,
    has_completed_10_overs,
    is_eligible_match,
)


class TestChase(unittest.TestCase):

    def setUp(self):
        """Create synthetic second-innings snapshot fixture for unit tests."""
        rows = []
        # Create 14 training seasons (2008-2021) with 2 matches each = 28 rows
        for year in range(2008, 2022):
            for m in [1, 2]:
                won = 1 if (m == 1 or year % 2 == 0) else 0
                batting_team = "Mumbai Indians" if m == 1 else "Kolkata Knight Riders"
                bowling_team = "Chennai Super Kings" if m == 1 else "Delhi Capitals"
                winner = batting_team if won == 1 else bowling_team
                target = 160 + m * 10
                current_score = 75 + won * 15
                wickets = 2 if won == 1 else 5

                rows.append({
                    "match_id": year * 10 + m,
                    "season": year,
                    "venue": "Wankhede Stadium" if m == 1 else "Eden Gardens",
                    "city": "Mumbai" if m == 1 else "Kolkata",
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "target": target,
                    "current_score": current_score,
                    "wickets": wickets,
                    "runs_last_5": 35 + won * 10,
                    "wickets_last_5": 1,
                    "runs_needed": target - current_score,
                    "balls_remaining": 60,
                    "required_run_rate": round((target - current_score) / 10.0, 2),
                    "current_run_rate": round(current_score / 10.0, 2),
                    "winner": winner,
                    "target_achieved": won,
                })

        # Create 3 test seasons (2022-2024) with 2 matches each = 6 rows
        for year in range(2022, 2025):
            for m in [1, 2]:
                won = 1 if m == 1 else 0
                batting_team = "Gujarat Titans" if m == 1 else "Lucknow Super Giants"  # Unseen teams
                bowling_team = "Rajasthan Royals"
                winner = batting_team if won == 1 else bowling_team
                target = 170 + m * 10
                current_score = 80 + won * 15
                wickets = 2 if won == 1 else 5

                rows.append({
                    "match_id": year * 10 + m,
                    "season": year,
                    "venue": "Ekana Stadium",  # Unseen venue
                    "city": "Lucknow",
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "target": target,
                    "current_score": current_score,
                    "wickets": wickets,
                    "runs_last_5": 38 + won * 10,
                    "wickets_last_5": 1,
                    "runs_needed": target - current_score,
                    "balls_remaining": 60,
                    "required_run_rate": round((target - current_score) / 10.0, 2),
                    "current_run_rate": round(current_score / 10.0, 2),
                    "winner": winner,
                    "target_achieved": won,
                })

        self.synthetic_df = pd.DataFrame(rows)

    def test_target_construction_and_binary_integrity(self):
        """Verify binary target: 1 when chasing team wins, 0 when defending team wins."""
        for _, row in self.synthetic_df.iterrows():
            if row["winner"] == row["batting_team"]:
                self.assertEqual(row["target_achieved"], 1)
            elif row["winner"] == row["bowling_team"]:
                self.assertEqual(row["target_achieved"], 0)
            self.assertIn(row["target_achieved"], [0, 1])

    def test_excluded_outcomes_and_shortened_matches(self):
        """Verify D/L matches, super overs, no-results, and rain-reduced matches are excluded."""
        dl_match = pd.Series({"method": "D/L", "super_over": "N", "result": "runs", "target_overs": 20})
        super_over_match = pd.Series({"method": None, "super_over": "Y", "result": "tie", "target_overs": 20})
        no_result_match = pd.Series({"method": None, "super_over": "N", "result": "no result", "target_overs": 20})
        rain_reduced_match = pd.Series({"method": None, "super_over": "N", "result": "wickets", "target_overs": 15})
        valid_match = pd.Series({"method": None, "super_over": "N", "result": "wickets", "target_overs": 20})

        self.assertFalse(is_eligible_match(dl_match))
        self.assertFalse(is_eligible_match(super_over_match))
        self.assertFalse(is_eligible_match(no_result_match))
        self.assertFalse(is_eligible_match(rain_reduced_match))
        self.assertTrue(is_eligible_match(valid_match))

    def test_target_leakage_prevention(self):
        """Verify winner, match result, and target are strictly separated from feature set X."""
        X, y = prepare_chase_features_and_target(self.synthetic_df)

        # Target column must NOT be in X
        self.assertNotIn(CHASE_TARGET_COLUMN, X.columns)
        self.assertEqual(y.name, CHASE_TARGET_COLUMN)

        # Post-match and tracking columns must NOT be in X
        self.assertNotIn("winner", X.columns)
        self.assertNotIn("match_id", X.columns)
        self.assertNotIn("season", X.columns)
        self.assertNotIn("city", X.columns)

        # X must contain only approved 10-over snapshot features
        expected_features = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
        self.assertEqual(list(X.columns), expected_features)

    def test_chronological_splitting_and_no_match_overlap(self):
        """Verify chronological split at 2021 and zero match overlap between partitions."""
        train_df, test_df = split_second_innings_data(self.synthetic_df, split_season=2021)

        self.assertEqual(len(train_df), 28)
        self.assertEqual(len(test_df), 6)
        self.assertTrue((train_df["season"] <= 2021).all())
        self.assertTrue((test_df["season"] >= 2022).all())

        overlap = set(train_df["match_id"]).intersection(set(test_df["match_id"]))
        self.assertEqual(len(overlap), 0)

    def test_unseen_categories_handling(self):
        """Verify models handle unseen teams and venues in test set without crashing."""
        train_df, test_df = split_second_innings_data(self.synthetic_df, split_season=2021)
        models = get_chase_models()
        rf = models["Random Forest"]

        pipeline = build_chase_pipeline(rf)
        X_train, y_train = prepare_chase_features_and_target(train_df)
        X_test, _ = prepare_chase_features_and_target(test_df)

        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]

        self.assertEqual(len(probs), len(test_df))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_probability_outputs_and_evaluation_metrics(self):
        """Verify probability bounds, Brier score, and classification metrics calculation."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.7])

        metrics = evaluate_chase_predictions(y_true, y_pred, y_prob)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Verify Brier score calculation: ((0.9-1)^2 + (0.1-0)^2 + (0.8-1)^2 + (0.7-0)^2) / 4
        # = (0.01 + 0.01 + 0.04 + 0.49) / 4 = 0.55 / 4 = 0.1375
        self.assertAlmostEqual(metrics["brier_score"], 0.1375, places=4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_real_dataset_chase_win_prediction(self):
        """Integration test on actual IPL dataset verifying split sizes and performance bounds."""
        output = run_chase_win_prediction()

        # Check train and test split sizes (826 train, 212 test)
        self.assertEqual(output["train_rows"], 826)
        self.assertEqual(output["test_rows"], 212)

        # Check train and test distributions
        train_dist = output["train_distribution"]
        test_dist = output["test_distribution"]
        self.assertEqual(train_dist[0], 370)
        self.assertEqual(train_dist[1], 456)
        self.assertEqual(test_dist[0], 110)
        self.assertEqual(test_dist[1], 102)

        models = output["models"]

        # Majority baseline in train is class 1 (456 > 370), so it predicts 1 for all test samples
        # Test accuracy should be exactly 102 / 212 = 0.4811
        baseline = models["Majority Baseline"]
        self.assertAlmostEqual(baseline["accuracy"], 102 / 212, places=3)
        self.assertEqual(baseline["confusion_matrix"], [[0, 110], [0, 102]])

        # Probabilistic ML models must substantially beat majority baseline
        for model_name in ["Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Gradient Boosting"]:
            self.assertIn(model_name, models)
            m = models[model_name]
            self.assertGreater(m["accuracy"], 0.70, f"{model_name} accuracy {m['accuracy']} <= 0.70")
            self.assertGreater(m["roc_auc"], 0.75, f"{model_name} ROC-AUC {m['roc_auc']} <= 0.75")
            self.assertLess(m["brier_score"], 0.20, f"{model_name} Brier score {m['brier_score']} >= 0.20")


if __name__ == "__main__":
    unittest.main()
