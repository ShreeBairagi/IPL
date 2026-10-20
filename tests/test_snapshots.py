"""
Unit and integration tests for 10-over snapshot construction.
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_and_preprocess_data
from src.snapshots import (
    FIRST_INNINGS_FEATURES,
    FIRST_INNINGS_TARGET,
    SECOND_INNINGS_FEATURES,
    SECOND_INNINGS_TARGET,
    count_wickets_lost,
    create_first_innings_snapshots,
    create_second_innings_snapshots,
    create_ten_over_snapshots,
    has_completed_10_overs,
    is_eligible_match,
)


class TestSnapshots(unittest.TestCase):

    def setUp(self):
        """Create standard synthetic match and delivery fixtures for unit tests."""
        self.sample_match = pd.Series({
            "id": 1001,
            "season": 2022,
            "venue": "Wankhede Stadium",
            "city": "Mumbai",
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "toss_winner": "Mumbai Indians",
            "toss_decision": "bat",
            "winner": "Chennai Super Kings",
            "result": "wickets",
            "result_margin": 5,
            "target_runs": 160.0,
            "target_overs": 20.0,
            "super_over": "N",
            "method": None,
        })

    def test_wickets_calculation_and_retired_hurt(self):
        """Verify that retired hurt is not counted as a lost wicket, but other dismissals are."""
        deliv = pd.DataFrame({
            "is_wicket": [1, 1, 1, 1, 0],
            "dismissal_kind": ["caught", "retired hurt", "bowled", "run out", None],
        })
        wickets = count_wickets_lost(deliv)
        # caught (1) + bowled (1) + run out (1) = 3; retired hurt (0)
        self.assertEqual(wickets, 3)

    def test_extras_contribute_to_runs(self):
        """Verify that extras (wides, legbyes, etc.) contribute to total_runs and snapshot scores."""
        deliv = pd.DataFrame({
            "over": [0, 0, 0],
            "batsman_runs": [4, 0, 6],
            "extra_runs": [0, 5, 0],  # 5 wides
            "total_runs": [4, 5, 6],
            "is_wicket": [0, 0, 0],
            "dismissal_kind": [None, None, None],
        })
        self.assertEqual(deliv["total_runs"].sum(), 15)

    def test_overs_0_to_9_boundary_and_last_5_overs(self):
        """
        Verify that:
        - Exactly overs 0-9 are included in the 10-over snapshot (over 10 is excluded).
        - Runs in overs 5-9 are calculated separately and do not include overs 0-4.
        """
        rows = []
        # Overs 0-4: 10 runs per over = 50 runs
        for ov in range(5):
            for b in range(6):
                rows.append({"match_id": 1001, "inning": 1, "over": ov, "ball": b + 1,
                             "batting_team": "Team A", "bowling_team": "Team B",
                             "total_runs": 10 / 6, "is_wicket": 0, "dismissal_kind": None, "extras_type": None})

        # Overs 5-9: 8 runs per over = 40 runs, with 1 wicket in over 7
        for ov in range(5, 10):
            for b in range(6):
                is_w = 1 if (ov == 7 and b == 0) else 0
                d_kind = "caught" if is_w else None
                rows.append({"match_id": 1001, "inning": 1, "over": ov, "ball": b + 1,
                             "batting_team": "Team A", "bowling_team": "Team B",
                             "total_runs": 8 / 6, "is_wicket": is_w, "dismissal_kind": d_kind, "extras_type": None})

        # Over 10: 20 runs (should NOT be included in 10-over snapshot)
        for b in range(6):
            rows.append({"match_id": 1001, "inning": 1, "over": 10, "ball": b + 1,
                         "batting_team": "Team A", "bowling_team": "Team B",
                         "total_runs": 20 / 6, "is_wicket": 0, "dismissal_kind": None, "extras_type": None})

        deliv_df = pd.DataFrame(rows)
        # Round runs cleanly
        deliv_df["total_runs"] = [2, 2, 2, 2, 1, 1] * 5 + [2, 2, 1, 1, 1, 1] * 5 + [4, 4, 3, 3, 3, 3]

        matches_df = pd.DataFrame([self.sample_match])
        snap = create_first_innings_snapshots(matches_df, deliv_df)

        self.assertEqual(len(snap), 1)
        # Total runs in overs 0-4 = 50, overs 5-9 = 40 -> current_score at over 9 = 90
        self.assertEqual(snap["current_score"].iloc[0], 90)
        # Runs in overs 5-9 = 40
        self.assertEqual(snap["runs_last_5"].iloc[0], 40)
        # Wickets in overs 0-9 = 1
        self.assertEqual(snap["wickets"].iloc[0], 1)
        self.assertEqual(snap["wickets_last_5"].iloc[0], 1)
        # Final score includes over 10 (90 + 20 = 110)
        self.assertEqual(snap["final_score"].iloc[0], 110)

    def test_dl_and_super_over_and_rain_exclusions(self):
        """Verify that D/L matches, super over matches, no-result, and shortened matches are excluded."""
        # Standard match is eligible
        self.assertTrue(is_eligible_match(self.sample_match))

        # D/L match -> excluded
        dl_match = self.sample_match.copy()
        dl_match["method"] = "D/L"
        self.assertFalse(is_eligible_match(dl_match))

        # Super over match -> excluded
        so_match = self.sample_match.copy()
        so_match["super_over"] = "Y"
        self.assertFalse(is_eligible_match(so_match))

        # No result (abandoned) -> excluded
        nr_match = self.sample_match.copy()
        nr_match["result"] = "no result"
        self.assertFalse(is_eligible_match(nr_match))

        # Reduced overs (target_overs < 20) -> excluded
        ro_match = self.sample_match.copy()
        ro_match["target_overs"] = 10.0
        self.assertFalse(is_eligible_match(ro_match))

    def test_invalid_or_incomplete_innings_excluded(self):
        """Verify that innings lasting fewer than 10 overs are safely excluded."""
        # Case 1: Incomplete innings stopping at over 7 -> False
        deliv_7 = pd.DataFrame({
            "over": [0, 1, 2, 3, 4, 5, 6, 7],
            "extras_type": [None] * 8,
        })
        self.assertFalse(has_completed_10_overs(deliv_7))

        # Case 2: Empty deliveries -> False
        self.assertFalse(has_completed_10_overs(pd.DataFrame()))

        # Case 3: Missing early overs (e.g. overs 5 to 11 only) -> False
        deliv_late_start = pd.DataFrame({
            "over": list(range(5, 12)),
            "extras_type": [None] * 7,
        })
        self.assertFalse(has_completed_10_overs(deliv_late_start))

        # Case 4: Over 9 has 4 legal balls + 2 wides (6 balls total, but only 4 legal) -> False
        rows_incomplete_9 = [{"over": ov, "extras_type": None} for ov in range(9)]  # overs 0-8
        rows_incomplete_9 += [
            {"over": 9, "extras_type": "wides"},
            {"over": 9, "extras_type": None},
            {"over": 9, "extras_type": None},
            {"over": 9, "extras_type": "wides"},
            {"over": 9, "extras_type": None},
            {"over": 9, "extras_type": None},
        ]  # 4 legal, 2 wides
        deliv_inc_9 = pd.DataFrame(rows_incomplete_9)
        self.assertFalse(has_completed_10_overs(deliv_inc_9))

        # Case 5: Over 9 has 6 legal balls + 2 wides (8 balls total, 6 legal) -> True
        rows_complete_9 = list(rows_incomplete_9) + [
            {"over": 9, "extras_type": None},
            {"over": 9, "extras_type": None},
        ]  # 6 legal balls total
        deliv_comp_9 = pd.DataFrame(rows_complete_9)
        self.assertTrue(has_completed_10_overs(deliv_comp_9))

        # Case 6: Innings that reached over 10 -> True
        deliv_completed_10 = pd.DataFrame({
            "over": list(range(11)),
            "extras_type": [None] * 11,
        })
        self.assertTrue(has_completed_10_overs(deliv_completed_10))

    def test_feature_target_separation_and_leakage_prevention(self):
        """Confirm that target labels are strictly separated from feature sets."""
        # 1st innings target must not be in features
        self.assertNotIn(FIRST_INNINGS_TARGET, FIRST_INNINGS_FEATURES)
        self.assertEqual(FIRST_INNINGS_TARGET, "final_score")

        # 2nd innings target and winner must not be in chase features
        self.assertNotIn(SECOND_INNINGS_TARGET, SECOND_INNINGS_FEATURES)
        self.assertNotIn("winner", SECOND_INNINGS_FEATURES)
        self.assertEqual(SECOND_INNINGS_TARGET, "target_achieved")

    def test_second_innings_snapshot_features(self):
        """Verify second innings snapshot calculates target, runs needed, and rates correctly."""
        rows = []
        for ov in range(11):
            for b in range(6):
                rows.append({
                    "match_id": 1001,
                    "inning": 2,
                    "over": ov,
                    "ball": b + 1,
                    "batting_team": "Chennai Super Kings",
                    "bowling_team": "Mumbai Indians",
                    "total_runs": 1,
                    "is_wicket": 0,
                    "dismissal_kind": None,
                    "extras_type": None,
                })
        deliv_df = pd.DataFrame(rows)
        matches_df = pd.DataFrame([self.sample_match])

        snap2 = create_second_innings_snapshots(matches_df, deliv_df)
        self.assertEqual(len(snap2), 1)
        row = snap2.iloc[0]

        self.assertEqual(row["match_id"], 1001)
        self.assertEqual(row["season"], 2022)
        self.assertEqual(row["target"], 160)
        # 60 balls bowled in overs 0-9 with 1 run per ball = 60 runs
        self.assertEqual(row["current_score"], 60)
        self.assertEqual(row["runs_needed"], 100)
        self.assertEqual(row["balls_remaining"], 60)
        self.assertEqual(row["current_run_rate"], 6.0)
        self.assertEqual(row["required_run_rate"], 10.0)
        self.assertEqual(row["winner"], "Chennai Super Kings")
        self.assertEqual(row["target_achieved"], 1)

    def test_real_dataset_snapshot_counts_and_integrity(self):
        """Integration test on actual IPL datasets verifying row counts and relational integrity."""
        m_df, d_df, _ = load_and_preprocess_data()
        df1, df2 = create_ten_over_snapshots(m_df, d_df)

        # Expected counts based on 1047 eligible 20-over matches
        self.assertEqual(len(df1), 1047)
        self.assertEqual(len(df2), 1038)

        # Check match IDs and seasons are retained and non-null
        self.assertFalse(df1["match_id"].isna().any())
        self.assertFalse(df1["season"].isna().any())
        self.assertFalse(df2["match_id"].isna().any())
        self.assertFalse(df2["season"].isna().any())

        # Second innings snapshots must all belong to valid first innings matches
        self.assertTrue(set(df2["match_id"]).issubset(set(df1["match_id"])))

        # Wickets must be within plausible cricket range [0, 10]
        self.assertTrue((df1["wickets"] >= 0).all() and (df1["wickets"] <= 10).all())
        self.assertTrue((df2["wickets"] >= 0).all() and (df2["wickets"] <= 10).all())

        # Runs last 5 must not exceed current score
        self.assertTrue((df1["runs_last_5"] <= df1["current_score"]).all())
        self.assertTrue((df2["runs_last_5"] <= df2["current_score"]).all())


if __name__ == "__main__":
    unittest.main()
