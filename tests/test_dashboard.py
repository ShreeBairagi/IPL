"""
Unit tests for dashboard helper functions and input validation.
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard_utils import (
    build_chase_input_df,
    build_first_innings_input_df,
    get_project_overview_data,
    get_teams_and_venues,
    validate_chase_inputs,
    validate_first_innings_inputs,
)
from src.regression import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.chase import CHASE_CATEGORICAL_FEATURES, CHASE_NUMERIC_FEATURES


class TestDashboard(unittest.TestCase):

    def test_first_innings_validation_rules(self):
        """Verify input validation catches boundary and semantic errors in first-innings inputs."""
        # Valid input
        valid, err = validate_first_innings_inputs(85, 2, 45, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertTrue(valid)
        self.assertIsNone(err)

        # Same team for batting and bowling
        valid, err = validate_first_innings_inputs(85, 2, 45, 1, "Mumbai Indians", "Mumbai Indians")
        self.assertFalse(valid)
        self.assertIn("different", err.lower())

        # Negative score
        valid, err = validate_first_innings_inputs(-10, 2, 45, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("negative", err.lower())

        # Wickets out of range (> 10)
        valid, err = validate_first_innings_inputs(85, 11, 45, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("10", err)

        # Runs in last 5 exceeding current score
        valid, err = validate_first_innings_inputs(80, 2, 85, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("cannot exceed", err.lower())

        # Wickets in last 5 exceeding total wickets
        valid, err = validate_first_innings_inputs(80, 2, 40, 3, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("cannot exceed", err.lower())

    def test_chase_validation_rules(self):
        """Verify input validation catches boundary and semantic errors in chase inputs."""
        # Valid input
        valid, err = validate_chase_inputs(170, 85, 2, 45, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertTrue(valid)
        self.assertIsNone(err)

        # Target <= 0
        valid, err = validate_chase_inputs(0, 85, 2, 45, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("positive", err.lower())

        # Runs in last 5 exceeding current score
        valid, err = validate_chase_inputs(170, 50, 2, 60, 1, "Mumbai Indians", "Chennai Super Kings")
        self.assertFalse(valid)
        self.assertIn("cannot exceed", err.lower())

    def test_build_first_innings_input_df(self):
        """Verify feature frame structure for first innings matches model expectations."""
        df = build_first_innings_input_df("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium", 85, 2, 45, 1)
        expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.assertEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["current_score"].iloc[0], 85)
        self.assertEqual(df["wickets"].iloc[0], 2)

    def test_build_chase_input_df_derived_features(self):
        """Verify chase feature frame correctly computes runs_needed, RRR, and CRR."""
        # Target: 180, Score at 10 overs: 80
        # runs_needed: 100, RRR: 10.0, CRR: 8.0
        df = build_chase_input_df("Kolkata Knight Riders", "Delhi Capitals", "Eden Gardens", 180, 80, 3, 40, 1)
        expected_cols = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
        self.assertEqual(list(df.columns), expected_cols)
        self.assertEqual(df["runs_needed"].iloc[0], 100)
        self.assertEqual(df["required_run_rate"].iloc[0], 10.0)
        self.assertEqual(df["current_run_rate"].iloc[0], 8.0)

    def test_teams_and_venues_retrieval(self):
        """Verify team and venue lists are non-empty, sorted, and free of redundant venue variants."""
        teams, venues = get_teams_and_venues()
        self.assertGreater(len(teams), 10)
        self.assertEqual(len(venues), 37)
        self.assertIn("Mumbai Indians", teams)
        self.assertIn("Chennai Super Kings", teams)

        # Check canonical venues are present
        self.assertIn("Wankhede Stadium", venues)
        self.assertIn("Eden Gardens", venues)
        self.assertIn("Arun Jaitley Stadium", venues)
        self.assertIn("Maharashtra Cricket Association Stadium", venues)

        # Check redundant/historical variants are NOT in dropdown
        self.assertNotIn("Wankhede Stadium, Mumbai", venues)
        self.assertNotIn("Eden Gardens, Kolkata", venues)
        self.assertNotIn("Feroz Shah Kotla", venues)
        self.assertNotIn("Subrata Roy Sahara Stadium", venues)
        self.assertNotIn("M.Chinnaswamy Stadium", venues)

    def test_build_input_dfs_normalize_venues(self):
        """Verify input DataFrame builders normalize raw or suffixed venue strings before inference."""
        df_first = build_first_innings_input_df("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium, Mumbai", 85, 2, 45, 1)
        self.assertEqual(df_first["venue"].iloc[0], "Wankhede Stadium")

        df_chase = build_chase_input_df("Kolkata Knight Riders", "Delhi Capitals", "Eden Gardens, Kolkata", 180, 80, 3, 40, 1)
        self.assertEqual(df_chase["venue"].iloc[0], "Eden Gardens")

    def test_project_overview_data_integrity(self):
        """Verify project overview statistics reflect actual dataset counts."""
        data = get_project_overview_data()
        self.assertEqual(data["raw_matches_count"], 1095)
        self.assertEqual(data["raw_deliveries_count"], 260920)
        self.assertEqual(data["first_innings_total"], 1047)
        self.assertEqual(data["first_innings_train"], 832)
        self.assertEqual(data["first_innings_test"], 215)
        self.assertEqual(data["second_innings_total"], 1038)
        self.assertEqual(data["second_innings_train"], 826)
        self.assertEqual(data["second_innings_test"], 212)


if __name__ == "__main__":
    unittest.main()
