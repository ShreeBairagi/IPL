"""
Unit and integration tests for IPL batter performance clustering.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batter_clustering import (
    DEFAULT_FEATURES,
    DEFAULT_K_RANGE,
    DEFAULT_MIN_BALLS,
    DEFAULT_RANDOM_STATE,
    calculate_batter_statistics,
    cluster_batters,
    evaluate_kmeans_clusters,
    filter_batters,
    run_batter_clustering,
    summarize_clusters,
)


class TestBatterClustering(unittest.TestCase):
    """Unit and integration tests for batter clustering module."""

    def setUp(self):
        """Create synthetic delivery records for controlled, isolated testing."""
        # Small synthetic deliveries dataset
        self.sample_deliveries = pd.DataFrame({
            "match_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "inning": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "batter": [
                "Batter A", "Batter A", "Batter A", "Batter A", "Batter B", "Batter B",
                "Batter A", "Batter A", "Batter B", "Batter B"
            ],
            "bowler": ["Bowler X"] * 10,
            "batsman_runs": [4, 6, 1, 0, 0, 1, 2, 4, 6, 1],
            "extra_runs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "total_runs": [4, 6, 1, 0, 0, 1, 2, 4, 6, 1],
            "extras_type": [None] * 10,
            "is_wicket": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            "player_dismissed": [None, None, None, "Batter A", None, None, None, None, None, "Batter B"],
            "dismissal_kind": [None, None, None, "caught", None, None, None, None, None, "bowled"],
        })

    def test_calculate_batter_statistics_basic(self):
        """Verify correct aggregation of runs, balls faced, innings, boundaries, and rates."""
        stats = calculate_batter_statistics(self.sample_deliveries)

        self.assertEqual(len(stats), 2)
        self.assertIn("batter", stats.columns)

        batter_a = stats[stats["batter"] == "Batter A"].iloc[0]
        # Match 1: 4, 6, 1, 0 (11 runs, 4 balls)
        # Match 2: 2, 4 (6 runs, 2 balls)
        # Total runs = 17, balls = 6, innings = 2, fours = 2, sixes = 1, dismissals = 1
        self.assertEqual(batter_a["total_runs"], 17)
        self.assertEqual(batter_a["balls_faced"], 6)
        self.assertEqual(batter_a["innings"], 2)
        self.assertEqual(batter_a["fours"], 2)
        self.assertEqual(batter_a["sixes"], 1)
        self.assertEqual(batter_a["dismissals"], 1)
        self.assertAlmostEqual(batter_a["batting_average"], 17.0, places=2)
        self.assertAlmostEqual(batter_a["strike_rate"], (17 / 6) * 100.0, places=1)

        batter_b = stats[stats["batter"] == "Batter B"].iloc[0]
        # Match 1: 0, 1 (1 run, 2 balls)
        # Match 2: 6, 1 (7 runs, 2 balls)
        # Total runs = 8, balls = 4, innings = 2, fours = 0, sixes = 1, dismissals = 1
        self.assertEqual(batter_b["total_runs"], 8)
        self.assertEqual(batter_b["balls_faced"], 4)
        self.assertEqual(batter_b["innings"], 2)
        self.assertEqual(batter_b["fours"], 0)
        self.assertEqual(batter_b["sixes"], 1)
        self.assertEqual(batter_b["dismissals"], 1)
        self.assertAlmostEqual(batter_b["batting_average"], 8.0, places=2)
        self.assertAlmostEqual(batter_b["strike_rate"], (8 / 4) * 100.0, places=1)

    def test_correct_handling_of_extras_and_balls_faced(self):
        """
        Verify cricket extras handling:
        - Wides: excluded from balls faced and do not credit batter runs.
        - No-balls: faced by batter, count as ball faced, batter runs credited.
        - Byes / Leg-byes: faced by batter, count as ball faced, extra_runs NOT credited to batter.
        """
        extras_deliveries = pd.DataFrame({
            "match_id": [1] * 5,
            "inning": [1] * 5,
            "batter": ["Player X"] * 5,
            "batsman_runs": [0, 2, 0, 0, 4],
            "extra_runs": [1, 1, 1, 1, 0],
            "extras_type": ["wides", "noballs", "byes", "legbyes", None],
            "player_dismissed": [None] * 5,
            "dismissal_kind": [None] * 5,
        })

        stats = calculate_batter_statistics(extras_deliveries)
        player_x = stats[stats["batter"] == "Player X"].iloc[0]

        # Wide delivery does not count as ball faced (4 balls faced total: noball, bye, legbye, legal ball)
        self.assertEqual(player_x["balls_faced"], 4)
        # Batter runs = 2 (noball shot) + 4 (normal boundary) = 6
        # Byes, legbyes, and wides do not credit batsman_runs
        self.assertEqual(player_x["total_runs"], 6)
        self.assertEqual(player_x["fours"], 1)
        # Strike rate: 6 runs off 4 balls = 150.0
        self.assertAlmostEqual(player_x["strike_rate"], 150.0, places=2)

    def test_dismissals_and_undefined_batting_averages(self):
        """
        Verify dismissal edge cases:
        - Batter with 0 dismissals (not out) has batting average explicitly set to NaN (not total_runs).
        - Retired hurt does NOT count as a dismissal under MCC Law 25.
        - Non-striker run out does NOT count as a dismissal for the facing batter.
        - Non-striker run out IS counted as a dismissal for the dismissed non-striker.
        - Batter with 0 balls faced has safe strike rate 0.0.
        """
        deliv = pd.DataFrame({
            "match_id": [1, 1, 1, 1],
            "inning": [1, 1, 1, 1],
            "batter": ["NotOut Player", "NotOut Player", "Striker", "Retired Player"],
            "non_striker": ["Partner A", "Partner A", "NonStriker Partner", "Partner B"],
            "batsman_runs": [10, 20, 5, 15],
            "extra_runs": [0, 0, 0, 0],
            "extras_type": [None, None, None, None],
            "player_dismissed": [None, None, "NonStriker Partner", "Retired Player"],
            "dismissal_kind": [None, None, "run out", "retired hurt"],
        })

        stats = calculate_batter_statistics(deliv)

        # 1. NotOut Player: 30 runs, 0 dismissals -> average = NaN (undefined, not total_runs)
        not_out = stats[stats["batter"] == "NotOut Player"].iloc[0]
        self.assertEqual(not_out["dismissals"], 0)
        self.assertEqual(not_out["total_runs"], 30)
        self.assertTrue(np.isnan(not_out["batting_average"]))

        # 2. Striker facing during non-striker run-out: dismissal belongs to NonStriker Partner, not Striker
        striker = stats[stats["batter"] == "Striker"].iloc[0]
        self.assertEqual(striker["dismissals"], 0)
        self.assertTrue(np.isnan(striker["batting_average"]))

        # 3. NonStriker Partner: dismissed via run-out, counted in stats
        non_striker = stats[stats["batter"] == "NonStriker Partner"].iloc[0]
        self.assertEqual(non_striker["dismissals"], 1)
        self.assertEqual(non_striker["innings"], 1)
        self.assertEqual(non_striker["balls_faced"], 0)
        self.assertEqual(non_striker["batting_average"], 0.0)

        # 4. Retired Player: retired hurt is not out under MCC Law 25
        retired = stats[stats["batter"] == "Retired Player"].iloc[0]
        self.assertEqual(retired["dismissals"], 0)
        self.assertTrue(np.isnan(retired["batting_average"]))

        # 5. Invariant: dismissals must not exceed innings
        self.assertTrue((stats["dismissals"] <= stats["innings"]).all())

    def test_non_striker_run_out(self):
        """Verify non-striker run-out is attributed to non-striker and not facing striker."""
        df = pd.DataFrame({
            "match_id": [1],
            "inning": [1],
            "batter": ["Facing Striker"],
            "non_striker": ["RunOut NonStriker"],
            "batsman_runs": [1],
            "extra_runs": [0],
            "extras_type": [None],
            "player_dismissed": ["RunOut NonStriker"],
            "dismissal_kind": ["run out"],
        })
        stats = calculate_batter_statistics(df)
        striker = stats[stats["batter"] == "Facing Striker"].iloc[0]
        non_striker = stats[stats["batter"] == "RunOut NonStriker"].iloc[0]

        self.assertEqual(striker["dismissals"], 0)
        self.assertEqual(non_striker["dismissals"], 1)
        self.assertEqual(non_striker["innings"], 1)
        self.assertEqual(non_striker["balls_faced"], 0)
        self.assertTrue((stats["dismissals"] <= stats["innings"]).all())

    def test_diamond_duck_dismissed_without_facing_ball(self):
        """Verify diamond duck (dismissed without facing a delivery) is preserved in stats."""
        df = pd.DataFrame({
            "match_id": [1, 2],
            "inning": [1, 1],
            "batter": ["Striker A", "Diamond Duck Player"],
            "non_striker": ["Diamond Duck Player", "Striker B"],
            "batsman_runs": [0, 4],
            "extra_runs": [0, 0],
            "extras_type": [None, None],
            "player_dismissed": ["Diamond Duck Player", None],
            "dismissal_kind": ["run out", None],
        })
        stats = calculate_batter_statistics(df)
        self.assertIn("Diamond Duck Player", stats["batter"].values)
        dd = stats[stats["batter"] == "Diamond Duck Player"].iloc[0]
        self.assertEqual(dd["total_runs"], 4)
        self.assertEqual(dd["balls_faced"], 1)
        self.assertEqual(dd["innings"], 2)  # Match 1 as non-striker, match 2 as batter
        self.assertEqual(dd["dismissals"], 1)
        self.assertEqual(dd["batting_average"], 4.0)
        self.assertTrue((stats["dismissals"] <= stats["innings"]).all())

    def test_non_striker_appearance_without_facing_deliveries(self):
        """Verify player who only appears at non-striker end has an inning counted and safe rates."""
        df = pd.DataFrame({
            "match_id": [10],
            "inning": [1],
            "batter": ["Sole Striker"],
            "non_striker": ["Passive NonStriker"],
            "batsman_runs": [6],
            "extra_runs": [0],
            "extras_type": [None],
            "player_dismissed": [None],
            "dismissal_kind": [None],
        })
        stats = calculate_batter_statistics(df)
        self.assertIn("Passive NonStriker", stats["batter"].values)
        pns = stats[stats["batter"] == "Passive NonStriker"].iloc[0]
        self.assertEqual(pns["innings"], 1)
        self.assertEqual(pns["balls_faced"], 0)
        self.assertEqual(pns["total_runs"], 0)
        self.assertEqual(pns["dismissals"], 0)
        self.assertTrue(np.isnan(pns["batting_average"]))
        self.assertEqual(pns["strike_rate"], 0.0)

    def test_retired_hurt_versus_retired_out(self):
        """Verify retired hurt is NOT a dismissal while retired out IS a dismissal under cricket law."""
        df = pd.DataFrame({
            "match_id": [1, 2],
            "inning": [1, 1],
            "batter": ["Hurt Player", "Tactical Player"],
            "non_striker": ["Partner A", "Partner B"],
            "batsman_runs": [25, 40],
            "extra_runs": [0, 0],
            "extras_type": [None, None],
            "player_dismissed": ["Hurt Player", "Tactical Player"],
            "dismissal_kind": ["retired hurt", "retired out"],
        })
        stats = calculate_batter_statistics(df)

        hurt = stats[stats["batter"] == "Hurt Player"].iloc[0]
        self.assertEqual(hurt["dismissals"], 0)
        self.assertTrue(np.isnan(hurt["batting_average"]))

        tactical = stats[stats["batter"] == "Tactical Player"].iloc[0]
        self.assertEqual(tactical["dismissals"], 1)
        self.assertEqual(tactical["batting_average"], 40.0)
        self.assertTrue((stats["dismissals"] <= stats["innings"]).all())

    def test_zero_dismissal_batting_average(self):
        """Verify batter with runs and zero dismissals has NaN batting average without total_runs substitution."""
        df = pd.DataFrame({
            "match_id": [1, 2],
            "inning": [1, 1],
            "batter": ["Invincible Batter", "Invincible Batter"],
            "non_striker": ["Partner A", "Partner B"],
            "batsman_runs": [45, 55],
            "extra_runs": [0, 0],
            "extras_type": [None, None],
            "player_dismissed": [None, None],
            "dismissal_kind": [None, None],
        })
        stats = calculate_batter_statistics(df)
        inv = stats[stats["batter"] == "Invincible Batter"].iloc[0]
        self.assertEqual(inv["total_runs"], 100)
        self.assertEqual(inv["dismissals"], 0)
        self.assertTrue(np.isnan(inv["batting_average"]))
        self.assertNotEqual(inv["batting_average"], 100.0)

    def test_dismissals_do_not_exceed_innings_invariant(self):
        """Verify that dismissals never exceed innings across historical deliveries."""
        df = pd.DataFrame({
            "match_id": [1, 2, 3],
            "inning": [1, 1, 1],
            "batter": ["Player A", "Player B", "Player A"],
            "non_striker": ["Player B", "Player A", "Player B"],
            "batsman_runs": [1, 2, 3],
            "extra_runs": [0, 0, 0],
            "extras_type": [None, None, None],
            "player_dismissed": ["Player B", "Player A", "Player B"],
            "dismissal_kind": ["run out", "bowled", "caught"],
        })
        stats = calculate_batter_statistics(df)
        self.assertTrue((stats["dismissals"] <= stats["innings"]).all())

    def test_zero_balls_faced_strike_rate(self):
        """Verify batter with only wide deliveries (0 balls faced) yields strike rate 0.0."""
        wides_only = pd.DataFrame({
            "match_id": [1],
            "inning": [1],
            "batter": ["Wide Only Batter"],
            "batsman_runs": [0],
            "extra_runs": [1],
            "extras_type": ["wides"],
            "player_dismissed": [None],
            "dismissal_kind": [None],
        })
        stats = calculate_batter_statistics(wides_only)
        batter = stats.iloc[0]
        self.assertEqual(batter["balls_faced"], 0)
        self.assertEqual(batter["strike_rate"], 0.0)
        self.assertFalse(np.isnan(batter["strike_rate"]))

    def test_minimum_sample_filtering(self):
        """Verify filtering batters by minimum balls faced and innings thresholds."""
        stats = pd.DataFrame({
            "batter": ["Player Low", "Player Mid", "Player High"],
            "total_runs": [50, 200, 500],
            "balls_faced": [40, 180, 420],
            "innings": [3, 12, 35],
            "dismissals": [2, 6, 14],
            "fours": [4, 18, 45],
            "sixes": [1, 5, 20],
            "batting_average": [25.0, 33.33, 35.71],
            "strike_rate": [125.0, 111.11, 119.05],
        })

        # Filter min_balls=200
        filtered_200 = filter_batters(stats, min_balls=200)
        self.assertEqual(len(filtered_200), 1)
        self.assertEqual(filtered_200.iloc[0]["batter"], "Player High")

        # Filter min_balls=100
        filtered_100 = filter_batters(stats, min_balls=100)
        self.assertEqual(len(filtered_100), 2)
        self.assertListEqual(filtered_100["batter"].tolist(), ["Player Mid", "Player High"])

        # Filter min_innings=10
        filtered_inn = filter_batters(stats, min_balls=0, min_innings=10)
        self.assertEqual(len(filtered_inn), 2)
        self.assertListEqual(filtered_inn["batter"].tolist(), ["Player Mid", "Player High"])

    def test_deterministic_cluster_assignments(self):
        """Verify clustering is strictly deterministic for fixed random_state and inputs."""
        rng = np.random.RandomState(42)
        n = 30
        synth_stats = pd.DataFrame({
            "batter": [f"Player_{i}" for i in range(n)],
            "batting_average": np.concatenate([rng.normal(35, 3, 15), rng.normal(18, 3, 15)]),
            "strike_rate": np.concatenate([rng.normal(140, 8, 15), rng.normal(110, 8, 15)]),
            "balls_faced": [300] * n,
            "total_runs": [600] * n,
            "innings": [25] * n,
            "dismissals": [15] * n,
            "fours": [50] * n,
            "sixes": [20] * n,
        })

        clustered1, model1, scaler1, scores1 = cluster_batters(
            synth_stats, n_clusters=2, random_state=42
        )
        clustered2, model2, scaler2, scores2 = cluster_batters(
            synth_stats, n_clusters=2, random_state=42
        )

        np.testing.assert_array_equal(clustered1["cluster"].values, clustered2["cluster"].values)
        self.assertEqual(scores1, scores2)
        np.testing.assert_array_almost_equal(model1.cluster_centers_, model2.cluster_centers_)

    def test_valid_cluster_count_selection_and_silhouette_score(self):
        """Verify evaluation of k=2..5 range and best_k selection using silhouette score."""
        rng = np.random.RandomState(100)
        n = 50
        # Two distinct groups
        group1 = rng.normal(loc=[38, 145], scale=[3, 5], size=(25, 2))
        group2 = rng.normal(loc=[18, 105], scale=[3, 5], size=(25, 2))
        data = np.vstack([group1, group2])

        synth_stats = pd.DataFrame(data, columns=["batting_average", "strike_rate"])
        synth_stats["batter"] = [f"Player_{i}" for i in range(n)]

        eval_res = evaluate_kmeans_clusters(
            synth_stats,
            features=["batting_average", "strike_rate"],
            k_range=(2, 5),
            random_state=42,
        )

        # Candidate k range (2, 3, 4, 5) evaluated
        self.assertSetEqual(set(eval_res["silhouette_scores"].keys()), {2, 3, 4, 5})
        for k, score in eval_res["silhouette_scores"].items():
            self.assertTrue(-1.0 <= score <= 1.0)

        # For two clearly separated clusters, best_k should be 2
        self.assertEqual(eval_res["best_k"], 2)
        self.assertGreater(eval_res["best_score"], 0.4)

    def test_small_sample_graceful_handling(self):
        """Verify evaluate_kmeans_clusters handles sample sizes smaller than k_range max."""
        tiny_df = pd.DataFrame({
            "batter": ["P1", "P2", "P3"],
            "batting_average": [20.0, 30.0, 40.0],
            "strike_rate": [110.0, 130.0, 150.0],
        })

        # k_range=(2, 5) on 3 samples: can only evaluate k=2
        eval_res = evaluate_kmeans_clusters(
            tiny_df,
            features=["batting_average", "strike_rate"],
            k_range=(2, 5),
            random_state=42,
        )
        self.assertIn(2, eval_res["silhouette_scores"])
        self.assertNotIn(3, eval_res["silhouette_scores"])
        self.assertNotIn(4, eval_res["silhouette_scores"])
        self.assertNotIn(5, eval_res["silhouette_scores"])

    def test_summarize_clusters(self):
        """Verify summarize_clusters computes aggregated metrics and identifies top players."""
        clustered = pd.DataFrame({
            "batter": ["Player A", "Player B", "Player C", "Player D"],
            "cluster": [0, 0, 1, 1],
            "total_runs": [1500, 1200, 400, 300],
            "balls_faced": [1100, 900, 350, 260],
            "innings": [50, 40, 25, 20],
            "dismissals": [40, 35, 20, 18],
            "fours": [120, 100, 30, 20],
            "sixes": [40, 30, 10, 5],
            "batting_average": [37.5, 34.29, 20.0, 16.67],
            "strike_rate": [136.36, 133.33, 114.29, 115.38],
            "boundary_pct": [58.67, 56.67, 45.0, 40.0],
        })

        summary = summarize_clusters(clustered)
        self.assertEqual(len(summary), 2)
        self.assertIn("mean_batting_average", summary.columns)
        self.assertIn("mean_strike_rate", summary.columns)
        self.assertIn("representative_batters", summary.columns)

        cluster_0 = summary[summary["cluster"] == 0].iloc[0]
        self.assertEqual(cluster_0["batter_count"], 2)
        self.assertAlmostEqual(cluster_0["mean_batting_average"], (37.5 + 34.29) / 2, places=1)
        self.assertIn("Player A", cluster_0["representative_batters"])

    def test_no_mutation_of_inputs_or_raw_dataset(self):
        """Verify input DataFrame is not mutated and raw CSV files are untouched."""
        raw_deliveries_path = Path("Datasets/deliveries_2008-2024.csv")
        raw_matches_path = Path("Datasets/matches_2008-2024.csv")

        self.assertTrue(raw_deliveries_path.exists())
        self.assertTrue(raw_matches_path.exists())

        mtime_deliv_before = raw_deliveries_path.stat().st_mtime
        mtime_matches_before = raw_matches_path.stat().st_mtime

        df_copy = self.sample_deliveries.copy()
        _ = calculate_batter_statistics(self.sample_deliveries)

        pd.testing.assert_frame_equal(df_copy, self.sample_deliveries)

        self.assertEqual(raw_deliveries_path.stat().st_mtime, mtime_deliv_before)
        self.assertEqual(raw_matches_path.stat().st_mtime, mtime_matches_before)

    def test_real_dataset_pipeline_integration(self):
        """Integration test on actual IPL dataset confirming end-to-end execution."""
        result = run_batter_clustering(
            matches_path="Datasets/matches_2008-2024.csv",
            deliveries_path="Datasets/deliveries_2008-2024.csv",
            max_season=2021,
            min_balls=200,
        )

        self.assertIn("clustered_df", result)
        self.assertIn("summary_df", result)
        self.assertIn("silhouette_scores", result)
        self.assertIn("chosen_k", result)

        clustered_df = result["clustered_df"]
        summary_df = result["summary_df"]

        # Verification of results
        self.assertGreater(len(clustered_df), 100)
        self.assertIn("cluster", clustered_df.columns)
        self.assertIn(result["chosen_k"], [2, 3, 4, 5])

        # Cluster labels should not use player names as features
        self.assertNotIn("batter", result["features"])
        self.assertNotIn("team", str(result["features"]))
        self.assertNotIn("winner", str(result["features"]))

        # Check silhouette score is positive and valid
        best_score = result["silhouette_scores"][result["chosen_k"]]
        self.assertGreater(best_score, 0.3)

        # Check summary structure
        self.assertEqual(len(summary_df), result["chosen_k"])
        self.assertGreater(summary_df["batter_count"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
