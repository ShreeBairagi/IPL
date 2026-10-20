"""
Unit and integration tests for data loading and preprocessing.
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    DataValidationError,
    REQUIRED_DELIVERIES_COLUMNS,
    REQUIRED_MATCHES_COLUMNS,
    TEAM_NAME_MAPPING,
    VENUE_NAME_MAPPING,
    clean_deliveries,
    clean_matches,
    load_and_preprocess_data,
    normalize_venue,
    validate_datasets,
)


class TestPreprocessing(unittest.TestCase):

    def test_whitespace_and_placeholder_cleaning(self):
        sample_df = pd.DataFrame({
            "col1  ": ["  value1  ", "NA  ", "   ", "valid"],
            "  col2": [10, 20, 30, 40],
        })
        from src.preprocessing import _strip_and_clean_strings

        cleaned = _strip_and_clean_strings(sample_df)

        self.assertListEqual(list(cleaned.columns), ["col1", "col2"])
        self.assertEqual(cleaned["col1"].iloc[0], "value1")
        self.assertTrue(pd.isna(cleaned["col1"].iloc[1]))
        self.assertTrue(pd.isna(cleaned["col1"].iloc[2]))
        self.assertEqual(cleaned["col1"].iloc[3], "valid")
        self.assertEqual(cleaned["col2"].iloc[0], 10)

    def test_team_name_standardization(self):
        matches_sample = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "team1": [
                "Delhi Daredevils",
                "Kings XI Punjab",
                "Royal Challengers Bangalore",
                "Rising Pune Supergiants",
            ],
            "team2": [
                "Mumbai Indians",
                "Chennai Super Kings",
                "Rajasthan Royals",
                "Kolkata Knight Riders",
            ],
            "toss_winner": [
                "Delhi Daredevils",
                "Kings XI Punjab",
                "Royal Challengers Bangalore",
                "Rising Pune Supergiants",
            ],
            "winner": [
                "Delhi Daredevils",
                "Kings XI Punjab",
                "Royal Challengers Bangalore",
                "Rising Pune Supergiants",
            ],
        })

        cleaned = clean_matches(matches_sample)

        expected_teams = [
            "Delhi Capitals",
            "Punjab Kings",
            "Royal Challengers Bengaluru",
            "Rising Pune Supergiant",
        ]
        self.assertListEqual(cleaned["team1"].tolist(), expected_teams)
        self.assertListEqual(cleaned["toss_winner"].tolist(), expected_teams)
        self.assertListEqual(cleaned["winner"].tolist(), expected_teams)
        self.assertEqual(cleaned["team2"].iloc[0], "Mumbai Indians")

    def test_deliveries_cleaning_and_id_type(self):
        deliv_sample = pd.DataFrame({
            "match_id ": [335982, 335982],
            "inning ": [1, 1],
            "batting_team                ": ["Delhi Daredevils   ", "Mumbai Indians   "],
            "bowling_team                ": ["Kings XI Punjab    ", "Delhi Daredevils "],
            "batter                  ": ["  SC Ganguly  ", "BB McCullum   "],
            "extras_type ": ["                ", "wides           "],
            "player_dismissed        ": ["NA                      ", "V Sehwag                "],
            "dismissal_kind        ": ["NA                    ", "bowled                "],
            "fielder ": ["NA                      ", "JH Kallis               "],
        })

        cleaned = clean_deliveries(deliv_sample)

        self.assertIn("match_id", cleaned.columns)
        self.assertEqual(cleaned["match_id"].dtype, "int64")
        self.assertEqual(cleaned["batting_team"].iloc[0], "Delhi Capitals")
        self.assertEqual(cleaned["bowling_team"].iloc[0], "Punjab Kings")
        self.assertEqual(cleaned["batter"].iloc[0], "SC Ganguly")
        self.assertEqual(cleaned["batter"].iloc[1], "BB McCullum")
        self.assertTrue(pd.isna(cleaned["extras_type"].iloc[0]))
        self.assertEqual(cleaned["extras_type"].iloc[1], "wides")
        self.assertTrue(pd.isna(cleaned["player_dismissed"].iloc[0]))
        self.assertEqual(cleaned["player_dismissed"].iloc[1], "V Sehwag")
        self.assertTrue(pd.isna(cleaned["dismissal_kind"].iloc[0]))
        self.assertEqual(cleaned["dismissal_kind"].iloc[1], "bowled")
        self.assertTrue(pd.isna(cleaned["fielder"].iloc[0]))
        self.assertEqual(cleaned["fielder"].iloc[1], "JH Kallis")

    def test_validation_missing_columns(self):
        m_df = pd.DataFrame({"id": [1], "season": [2008]})
        d_df = pd.DataFrame({"match_id": [1], "inning": [1]})

        with self.assertRaises(DataValidationError) as ctx:
            validate_datasets(m_df, d_df)
        self.assertIn("missing required columns", str(ctx.exception).lower())

    def test_validation_duplicate_match_id(self):
        m_df = pd.DataFrame({col: [1, 1] for col in REQUIRED_MATCHES_COLUMNS})
        d_df = pd.DataFrame({col: [1, 1] for col in REQUIRED_DELIVERIES_COLUMNS})

        with self.assertRaises(DataValidationError) as ctx:
            validate_datasets(m_df, d_df)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_validation_missing_match_id(self):
        m_df = pd.DataFrame({col: [1, None] for col in REQUIRED_MATCHES_COLUMNS})
        d_df = pd.DataFrame({col: [1, 1] for col in REQUIRED_DELIVERIES_COLUMNS})

        with self.assertRaises(DataValidationError) as ctx:
            validate_datasets(m_df, d_df)
        self.assertIn("missing (null)", str(ctx.exception).lower())

    def test_validation_unmatched_delivery_id(self):
        m_df = pd.DataFrame({col: [100] for col in REQUIRED_MATCHES_COLUMNS})
        d_df = pd.DataFrame({col: [999] for col in REQUIRED_DELIVERIES_COLUMNS})

        with self.assertRaises(DataValidationError) as ctx:
            validate_datasets(m_df, d_df)
        self.assertIn("do not exist in matches", str(ctx.exception).lower())

    def test_full_pipeline_on_real_datasets(self):
        m_df, d_df, audit = load_and_preprocess_data()

        # Check row preservation
        self.assertEqual(audit["rows_matches_before"], 1095)
        self.assertEqual(audit["rows_matches_after"], 1095)
        self.assertEqual(audit["rows_deliveries_before"], 260920)
        self.assertEqual(audit["rows_deliveries_after"], 260920)

        # Check unmatched IDs
        self.assertEqual(audit["unmatched_delivery_match_ids"], 0)

        # Check teams count before and after
        self.assertEqual(len(audit["teams_matches_before"]), 19)
        self.assertEqual(len(audit["teams_matches_after"]), 15)
        self.assertEqual(len(audit["teams_deliveries_before"]), 19)
        self.assertEqual(len(audit["teams_deliveries_after"]), 15)

        # Check no whitespace in column names
        for col in m_df.columns:
            self.assertEqual(col, col.strip())
        for col in d_df.columns:
            self.assertEqual(col, col.strip())

        # Check ID data types
        self.assertEqual(m_df["id"].dtype, "int64")
        self.assertEqual(d_df["match_id"].dtype, "int64")

        # Check venue normalization on real dataset
        raw_venues_count = m_df["venue"].nunique()
        self.assertEqual(raw_venues_count, 37)

    def test_normalize_venue_approved_mappings(self):
        """Verify all approved mappings produce expected canonical venue names."""
        # City suffix removals & formatting
        self.assertEqual(normalize_venue("Eden Gardens, Kolkata"), "Eden Gardens")
        self.assertEqual(normalize_venue("Wankhede Stadium, Mumbai"), "Wankhede Stadium")
        self.assertEqual(normalize_venue("M.Chinnaswamy Stadium"), "M Chinnaswamy Stadium")
        self.assertEqual(normalize_venue("M Chinnaswamy Stadium, Bengaluru"), "M Chinnaswamy Stadium")
        self.assertEqual(normalize_venue("MA Chidambaram Stadium"), "MA Chidambaram Stadium, Chepauk")
        self.assertEqual(normalize_venue("MA Chidambaram Stadium, Chepauk, Chennai"), "MA Chidambaram Stadium, Chepauk")
        self.assertEqual(normalize_venue("Rajiv Gandhi International Stadium, Uppal"), "Rajiv Gandhi International Stadium")
        self.assertEqual(normalize_venue("Rajiv Gandhi International Stadium, Uppal, Hyderabad"), "Rajiv Gandhi International Stadium")
        self.assertEqual(normalize_venue("Sawai Mansingh Stadium, Jaipur"), "Sawai Mansingh Stadium")
        self.assertEqual(normalize_venue("Brabourne Stadium, Mumbai"), "Brabourne Stadium")
        self.assertEqual(normalize_venue("Dr DY Patil Sports Academy, Mumbai"), "Dr DY Patil Sports Academy")
        self.assertEqual(normalize_venue("Maharashtra Cricket Association Stadium, Pune"), "Maharashtra Cricket Association Stadium")
        self.assertEqual(normalize_venue("Arun Jaitley Stadium, Delhi"), "Arun Jaitley Stadium")
        self.assertEqual(normalize_venue("Zayed Cricket Stadium, Abu Dhabi"), "Sheikh Zayed Stadium")
        self.assertEqual(normalize_venue("Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"), "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium")
        self.assertEqual(normalize_venue("Himachal Pradesh Cricket Association Stadium, Dharamsala"), "Himachal Pradesh Cricket Association Stadium")
        self.assertEqual(normalize_venue("Barsapara Cricket Stadium, Guwahati"), "Barsapara Cricket Stadium")
        self.assertEqual(normalize_venue("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"), "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium")
        self.assertEqual(normalize_venue("Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"), "Maharaja Yadavindra Singh International Cricket Stadium")
        self.assertEqual(normalize_venue("Vidarbha Cricket Association Stadium, Jamtha"), "Vidarbha Cricket Association Stadium")
        self.assertEqual(normalize_venue("Narendra Modi Stadium, Ahmedabad"), "Narendra Modi Stadium")

        # Approved historical renamings
        self.assertEqual(normalize_venue("Feroz Shah Kotla"), "Arun Jaitley Stadium")
        self.assertEqual(normalize_venue("Subrata Roy Sahara Stadium"), "Maharashtra Cricket Association Stadium")

        # Kept distinct by decision
        self.assertEqual(normalize_venue("Sardar Patel Stadium, Motera"), "Sardar Patel Stadium, Motera")

    def test_normalize_venue_invariance(self):
        """Verify that equivalent venue variants all resolve to identical canonical strings."""
        groups = [
            (["Eden Gardens", "Eden Gardens, Kolkata"], "Eden Gardens"),
            (["Wankhede Stadium", "Wankhede Stadium, Mumbai"], "Wankhede Stadium"),
            (["M Chinnaswamy Stadium", "M.Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru"], "M Chinnaswamy Stadium"),
            (["MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium", "MA Chidambaram Stadium, Chepauk, Chennai"], "MA Chidambaram Stadium, Chepauk"),
            (["Rajiv Gandhi International Stadium", "Rajiv Gandhi International Stadium, Uppal", "Rajiv Gandhi International Stadium, Uppal, Hyderabad"], "Rajiv Gandhi International Stadium"),
            (["Punjab Cricket Association Stadium, Mohali", "Punjab Cricket Association IS Bindra Stadium, Mohali", "Punjab Cricket Association IS Bindra Stadium", "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"], "Punjab Cricket Association IS Bindra Stadium"),
            (["Feroz Shah Kotla", "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi"], "Arun Jaitley Stadium"),
            (["Maharashtra Cricket Association Stadium", "Maharashtra Cricket Association Stadium, Pune", "Subrata Roy Sahara Stadium"], "Maharashtra Cricket Association Stadium"),
        ]
        for variants, expected_canonical in groups:
            canonical_set = {normalize_venue(v) for v in variants}
            self.assertEqual(len(canonical_set), 1, f"Expected 1 canonical venue for {variants}, got {canonical_set}")
            self.assertEqual(canonical_set.pop(), expected_canonical)

    def test_normalize_venue_distinct_grounds_preserved(self):
        """Verify physically distinct grounds within the same metropolitan region are never merged."""
        # Mumbai stadiums
        mumbai_grounds = {
            normalize_venue("Wankhede Stadium, Mumbai"),
            normalize_venue("Brabourne Stadium, Mumbai"),
            normalize_venue("Dr DY Patil Sports Academy, Mumbai"),
        }
        self.assertEqual(len(mumbai_grounds), 3)

        # Mohali vs Mullanpur
        punjab_grounds = {
            normalize_venue("Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"),
            normalize_venue("Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"),
        }
        self.assertEqual(len(punjab_grounds), 2)

        # Motera vs Modi stadium
        ahmedabad_grounds = {
            normalize_venue("Sardar Patel Stadium, Motera"),
            normalize_venue("Narendra Modi Stadium, Ahmedabad"),
        }
        self.assertEqual(len(ahmedabad_grounds), 2)

    def test_normalize_venue_missing_and_unknown_handling(self):
        """Verify None, NaN, and unmapped venue names are preserved safely without guessing."""
        self.assertIsNone(normalize_venue(None))
        self.assertTrue(pd.isna(normalize_venue(float("nan"))))
        self.assertEqual(normalize_venue(""), "")
        # Unknown/unmapped venue remains unchanged
        self.assertEqual(normalize_venue("Melbourne Cricket Ground"), "Melbourne Cricket Ground")
        self.assertEqual(normalize_venue("Lord's Cricket Ground"), "Lord's Cricket Ground")

    def test_clean_matches_venue_normalization(self):
        """Verify clean_matches standardizes venue column in a sample DataFrame."""
        sample_df = pd.DataFrame({
            "id": [1, 2, 3],
            "venue": [" Eden Gardens, Kolkata ", "Feroz Shah Kotla", "Wankhede Stadium, Mumbai"],
        })
        cleaned = clean_matches(sample_df)
        expected = ["Eden Gardens", "Arun Jaitley Stadium", "Wankhede Stadium"]
        self.assertListEqual(cleaned["venue"].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
