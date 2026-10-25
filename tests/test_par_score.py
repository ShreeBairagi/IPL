"""
Unit tests for stadium-specific par score computation and classification.

Validates:
1. Par score calculation for known stadiums with sufficient data.
2. Low-sample stadiums and Bayesian shrinkage towards competition average.
3. Graceful fallback for unseen, new, or missing stadium inputs.
4. Strict chronological leakage prevention (training matches <= 2021 only).
5. Conditional classification logic (Low, Par, High) and boundary precision.
6. Output dictionary schema and data types.
7. End-to-end integration with trained regression pipelines.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.par_score import (
    DEFAULT_PAR_FILE,
    FALLBACK_VENUE_KEY,
    classify_stadium_score,
    compute_stadium_par_scores,
    load_stadium_par_scores,
    predict_score_with_stadium_par,
)
from src.preprocessing import load_and_preprocess_data, normalize_venue
from src.regression import (
    build_regression_pipeline,
    prepare_features_and_target,
    split_first_innings_data,
)
from src.snapshots import create_first_innings_snapshots


class TestStadiumParScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches_df, cls.deliveries_df, _ = load_and_preprocess_data()
        cls.snapshots_df = create_first_innings_snapshots(cls.matches_df, cls.deliveries_df)
        cls.stats_df = load_stadium_par_scores()

    def test_par_scores_file_structure(self):
        """Saved artifact contains required statistical columns and non-empty rows."""
        self.assertFalse(self.stats_df.empty)
        required_cols = [
            "venue",
            "match_count",
            "median_score",
            "mean_score",
            "std_score",
            "p33_score",
            "p66_score",
            "recent_mean_score",
            "par_score",
            "threshold_low",
            "threshold_high",
        ]
        for col in required_cols:
            self.assertIn(col, self.stats_df.columns, f"Missing required column {col} in par scores artifact.")

    def test_competition_average_fallback_row_present(self):
        """Artifact includes the competition fallback row matching global training tertiles."""
        comp_rows = self.stats_df[self.stats_df["venue"] == FALLBACK_VENUE_KEY]
        self.assertEqual(len(comp_rows), 1)
        comp = comp_rows.iloc[0]

        # Training set <= 2021 has 832 eligible matches with tertiles 152.0 and 175.0
        self.assertEqual(comp["match_count"], 832)
        self.assertEqual(comp["threshold_low"], 152.0)
        self.assertEqual(comp["threshold_high"], 175.0)
        self.assertEqual(comp["par_score"], 163.5)

    def test_known_high_scoring_stadium(self):
        """M Chinnaswamy Stadium reflects high scoring conditions."""
        chinnaswamy_rows = self.stats_df[self.stats_df["venue"] == "M Chinnaswamy Stadium"]
        self.assertEqual(len(chinnaswamy_rows), 1)
        row = chinnaswamy_rows.iloc[0]

        # High scoring venue has par score > 170 and high threshold > 180
        self.assertGreater(row["par_score"], 170.0)
        self.assertGreater(row["threshold_high"], 184.0)

        # A score of 180 at Chinnaswamy is typical (Par), not High
        eval_180 = classify_stadium_score(180, "M Chinnaswamy Stadium", self.stats_df)
        self.assertEqual(eval_180["classification"], "Par")
        self.assertFalse(eval_180["is_fallback"])

        # A score of 190 at Chinnaswamy is High
        eval_190 = classify_stadium_score(190, "M Chinnaswamy Stadium", self.stats_df)
        self.assertEqual(eval_190["classification"], "High")

    def test_known_low_scoring_stadium(self):
        """Chepauk reflects spin/lower scoring conditions relative to Chinnaswamy."""
        chepauk_rows = self.stats_df[self.stats_df["venue"] == "MA Chidambaram Stadium, Chepauk"]
        self.assertEqual(len(chepauk_rows), 1)
        row = chepauk_rows.iloc[0]

        # Par score at Chepauk should be noticeably lower than Chinnaswamy
        self.assertLess(row["par_score"], 165.0)

        # 180 runs at Chepauk is a commanding High score
        eval_chepauk = classify_stadium_score(180, "MA Chidambaram Stadium, Chepauk", self.stats_df)
        self.assertEqual(eval_chepauk["classification"], "High")

        # 145 runs at Chepauk is Low
        eval_low = classify_stadium_score(145, "MA Chidambaram Stadium, Chepauk", self.stats_df)
        self.assertEqual(eval_low["classification"], "Low")

    def test_rare_and_low_sample_stadium_shrinkage(self):
        """Stadiums with small sample sizes shrink towards the competition baseline."""
        # OUTsurance Oval only hosted 2 matches
        oval_rows = self.stats_df[self.stats_df["venue"] == "OUTsurance Oval"]
        self.assertEqual(len(oval_rows), 1)
        row = oval_rows.iloc[0]

        self.assertEqual(row["match_count"], 2)
        # Raw median was 135.0, but shrunk par score must be higher due to competition mean shrinkage
        self.assertGreater(row["par_score"], 145.0)
        self.assertLess(row["par_score"], 163.5)

    def test_unseen_and_none_venue_graceful_fallback(self):
        """Unseen venues or None input fall back gracefully to competition averages."""
        # Case 1: Brand new venue (e.g., Ekana Stadium in Lucknow, absent before 2022)
        unseen_eval = classify_stadium_score(
            160.0,
            "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
            self.stats_df,
        )
        self.assertTrue(unseen_eval["is_fallback"])
        self.assertEqual(unseen_eval["stadium_par_score"], 163.5)
        self.assertEqual(unseen_eval["classification"], "Par")
        self.assertEqual(unseen_eval["difference_from_par"], -3.5)

        # Case 2: None venue
        none_eval = classify_stadium_score(145.0, None, self.stats_df)
        self.assertTrue(none_eval["is_fallback"])
        self.assertEqual(none_eval["classification"], "Low")

    def test_chronological_leakage_prevention(self):
        """Calculating par scores with max_season=2021 strictly excludes 2022-2024 matches."""
        computed = compute_stadium_par_scores(
            self.matches_df,
            self.deliveries_df,
            max_season=2021,
        )
        total_matches = computed[computed["venue"] == FALLBACK_VENUE_KEY]["match_count"].iloc[0]

        # Verify against snapshots with season <= 2021
        train_snapshots = self.snapshots_df[self.snapshots_df["season"] <= 2021]
        self.assertEqual(total_matches, len(train_snapshots))

        # Test set matches (season >= 2022) must not be counted
        test_snapshots = self.snapshots_df[self.snapshots_df["season"] >= 2022]
        self.assertGreater(len(test_snapshots), 200)
        self.assertNotEqual(total_matches, len(self.snapshots_df))

    def test_classification_boundary_precision(self):
        """Exact boundary tests: score < low -> Low, between bounds -> Par, > high -> High."""
        mock_stats = pd.DataFrame([{
            "venue": "Test Ground",
            "match_count": 20,
            "median_score": 160.0,
            "mean_score": 160.0,
            "std_score": 15.0,
            "p33_score": 150.0,
            "p66_score": 170.0,
            "recent_mean_score": 160.0,
            "par_score": 160.0,
            "threshold_low": 150.0,
            "threshold_high": 170.0,
        }, {
            "venue": FALLBACK_VENUE_KEY,
            "match_count": 100,
            "median_score": 160.0,
            "mean_score": 160.0,
            "std_score": 15.0,
            "p33_score": 150.0,
            "p66_score": 170.0,
            "recent_mean_score": 160.0,
            "par_score": 160.0,
            "threshold_low": 150.0,
            "threshold_high": 170.0,
        }])

        self.assertEqual(classify_stadium_score(149.9, "Test Ground", mock_stats)["classification"], "Low")
        self.assertEqual(classify_stadium_score(150.0, "Test Ground", mock_stats)["classification"], "Par")
        self.assertEqual(classify_stadium_score(160.0, "Test Ground", mock_stats)["classification"], "Par")
        self.assertEqual(classify_stadium_score(170.0, "Test Ground", mock_stats)["classification"], "Par")
        self.assertEqual(classify_stadium_score(170.1, "Test Ground", mock_stats)["classification"], "High")

    def test_end_to_end_regression_integration(self):
        """Integrates with existing regression pipeline without altering predictions."""
        from sklearn.linear_model import Ridge

        train_df, _ = split_first_innings_data(self.snapshots_df, split_season=2021)
        X_train, y_train = prepare_features_and_target(train_df)

        pipeline = build_regression_pipeline(Ridge(alpha=1.0))
        pipeline.fit(X_train, y_train)

        result = predict_score_with_stadium_par(
            pipeline=pipeline,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            venue="Wankhede Stadium",
            current_score=95,
            wickets=1,
            runs_last_5=50,
            wickets_last_5=0,
            stats_df=self.stats_df,
        )

        # Expected output structure
        self.assertIn("predicted_score", result)
        self.assertIn("stadium_par_score", result)
        self.assertIn("difference_from_par", result)
        self.assertIn("classification", result)

        self.assertIsInstance(result["predicted_score"], float)
        self.assertIsInstance(result["stadium_par_score"], float)
        self.assertIsInstance(result["difference_from_par"], float)
        self.assertIn(result["classification"], ["Low", "Par", "High"])

        # Mathematical consistency
        expected_diff = round(result["predicted_score"] - result["stadium_par_score"], 1)
        self.assertEqual(result["difference_from_par"], expected_diff)


if __name__ == "__main__":
    unittest.main()
