"""
Automated Streamlit dashboard UI tests using streamlit.testing.v1.AppTest.

Tests the interactive IPL Match Intelligence Lab interface across:
1. Application startup and section navigation
2. Project Overview layout and dynamic metrics
3. First-Innings Final Score Prediction, classification, and validation
4. Chase-Win Probability estimation, situation metrics, and boundaries
5. Held-Out Model Evaluation tables and missing-metric handling
6. Regression, caching, and order repeatability
"""
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_PATH = str(PROJECT_ROOT / "app.py")

from src.dashboard_utils import (
    get_project_overview_data,
    load_all_models_and_metrics,
)


class TestAppStartup(unittest.TestCase):
    """Test scenario 1: Application startup, error freedom, and section navigation."""

    def test_startup_renders_without_exceptions(self):
        """App initializes cleanly without uncaught exceptions or import failures."""
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, f"Uncaught exceptions at startup: {at.exception}")
        # Verify app title rendered
        titles = [t.value for t in at.title]
        self.assertTrue(any("IPL Match Intelligence Lab" in t for t in titles))

    def test_all_four_sections_render_without_uncaught_exceptions(self):
        """All sidebar navigation options render without exceptions."""
        expected_sections = [
            "1. Project Overview",
            "2. Final Score Prediction & Class",
            "3. Chase-Win Probability",
            "4. Model Evaluation",
            "5. Batter Performance Clustering",
        ]
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        radio = at.sidebar.radio[0]
        self.assertEqual(list(radio.options), expected_sections)

        for sec in expected_sections:
            at.sidebar.radio[0].set_value(sec).run()
            self.assertEqual(
                len(at.exception),
                0,
                f"Uncaught exception when rendering section '{sec}': {at.exception}",
            )

    def test_no_missing_file_or_import_errors(self):
        """Verify no missing file or module import errors occur during execution."""
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        # Verify no error banners are present on fresh launch
        self.assertEqual(len(at.error), 0, f"Unexpected error messages: {[e.value for e in at.error]}")


class TestProjectOverviewSection(unittest.TestCase):
    """Test scenario 2: Project Overview layout, headings, and dynamic metrics."""

    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.at.sidebar.radio[0].set_value("1. Project Overview").run()

    def test_overview_headings_and_structure(self):
        """Expected overview headers and subheaders are rendered."""
        headers = [h.value for h in self.at.header]
        subheaders = [sh.value for sh in self.at.subheader]

        self.assertIn("Project Overview & Methodology", headers)
        self.assertIn("Strict Chronological Partitioning", subheaders)
        self.assertIn("Dataset & Snapshot Counts", subheaders)

    def test_overview_metrics_sourced_from_project_code(self):
        """Displayed counts match project code data (not hardcoded arbitrary values)."""
        expected = get_project_overview_data()

        metric_dict = {m.label: m.value for m in self.at.metric}
        self.assertIn("Raw Deliveries", metric_dict)
        self.assertIn("Raw Matches", metric_dict)
        self.assertIn("Cleaned 20-Over Matches", metric_dict)

        self.assertEqual(metric_dict["Raw Deliveries"], f"{expected['raw_deliveries_count']:,}")
        self.assertEqual(metric_dict["Raw Matches"], f"{expected['raw_matches_count']:,}")
        self.assertEqual(metric_dict["Cleaned 20-Over Matches"], f"{expected['cleaned_matches_count']:,}")

        # Check snapshot breakdown numbers in markdown / write elements
        all_text = " ".join([m.value for m in self.at.markdown] + [w.value for w in self.at.info])
        # Markdown strings also include write calls in AppTest
        all_text += " ".join(str(elem.value) for elem in getattr(self.at, "write", []))

        # Check that the test and train split seasons appear
        self.assertIn(expected["train_seasons_range"], all_text)
        self.assertIn(expected["test_seasons_range"], all_text)


class TestFinalScorePredictionSection(unittest.TestCase):
    """Test scenario 3: Final Score Prediction, classification, and validation."""

    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.at.sidebar.radio[0].set_value("2. Final Score Prediction & Class").run()

    def test_input_widgets_exist_and_accept_valid_values(self):
        """All expected selectbox, number_input, and button widgets are present."""
        selectbox_labels = [s.label for s in self.at.selectbox]
        self.assertIn("Batting Team (1st Innings)", selectbox_labels)
        self.assertIn("Bowling Team", selectbox_labels)
        self.assertIn("Venue / Stadium", selectbox_labels)
        self.assertIn("Regression Estimator", selectbox_labels)

        number_labels = [n.label for n in self.at.number_input]
        self.assertIn("Score at 10 Overs", number_labels)
        self.assertIn("Wickets Lost (0–10)", number_labels)
        self.assertIn("Runs in Overs 5–9", number_labels)
        self.assertIn("Wickets in Overs 5–9", number_labels)

        button_labels = [b.label for b in self.at.button]
        self.assertIn("Predict Final Score & Class", button_labels)

        # Verify Regression Estimator choices
        reg_sb = [s for s in self.at.selectbox if s.label == "Regression Estimator"][0]
        for expected_model in ["Ridge Regression", "Random Forest", "Linear Regression", "Gradient Boosting"]:
            self.assertIn(expected_model, reg_sb.options)

    def test_valid_submission_displays_score_and_classification(self):
        """Submitting valid inputs displays numerical score and score classification."""
        # Click submit button
        submit_btn = [b for b in self.at.button if b.label == "Predict Final Score & Class"][0]
        submit_btn.click().run()

        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(len(self.at.error), 0)

        # Check success banner
        success_msgs = [s.value for s in self.at.success]
        self.assertTrue(any("Prediction Generated" in m for m in success_msgs))

        # Check metrics
        metric_dict = {m.label: m.value for m in self.at.metric}

        # Find predicted score metric (label includes estimator name)
        score_metric = next(v for k, v in metric_dict.items() if "Predicted Final Score" in k)
        self.assertTrue(re.match(r"^\d+(\.\d+)? runs$", score_metric), f"Invalid score format: {score_metric}")
        score_val = float(score_metric.replace(" runs", ""))
        self.assertGreaterEqual(score_val, 50.0)
        self.assertLessEqual(score_val, 320.0)

        # Find predicted class metric
        class_metric = next(v for k, v in metric_dict.items() if "Predicted Score Class" in k)
        self.assertIn(class_metric, ["Low", "Medium", "High"])

    def test_changing_inputs_updates_prediction(self):
        """Changing input values updates the predicted score appropriately."""
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        wickets_input = [n for n in self.at.number_input if n.label == "Wickets Lost (0–10)"][0]
        runs_last_5 = [n for n in self.at.number_input if n.label == "Runs in Overs 5–9"][0]
        wickets_last_5 = [n for n in self.at.number_input if n.label == "Wickets in Overs 5–9"][0]
        submit_btn = [b for b in self.at.button if b.label == "Predict Final Score & Class"][0]

        # Case 1: Low score state
        score_input.set_value(50)
        wickets_input.set_value(5)
        runs_last_5.set_value(20)
        wickets_last_5.set_value(2)
        submit_btn.click().run()

        metrics_1 = {m.label: m.value for m in self.at.metric}
        score_metric_1 = next(v for k, v in metrics_1.items() if "Predicted Final Score" in k)
        val_1 = float(score_metric_1.replace(" runs", ""))

        # Case 2: High score state
        score_input.set_value(120)
        wickets_input.set_value(1)
        runs_last_5.set_value(55)
        wickets_last_5.set_value(0)
        submit_btn.click().run()

        metrics_2 = {m.label: m.value for m in self.at.metric}
        score_metric_2 = next(v for k, v in metrics_2.items() if "Predicted Final Score" in k)
        val_2 = float(score_metric_2.replace(" runs", ""))

        # A dominant batting performance must produce a strictly higher projected score
        self.assertGreater(val_2, val_1, f"Expected {val_2} > {val_1}")

    def test_semantic_invalid_inputs_display_validation_errors(self):
        """Invalid inputs surface understandable error messages and halt prediction."""
        batting_team = [s for s in self.at.selectbox if s.label == "Batting Team (1st Innings)"][0]
        bowling_team = [s for s in self.at.selectbox if s.label == "Bowling Team"][0]
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        runs_last_5 = [n for n in self.at.number_input if n.label == "Runs in Overs 5–9"][0]
        submit_btn = [b for b in self.at.button if b.label == "Predict Final Score & Class"][0]

        # Sub-case A: Same team selected for batting and bowling
        first_team = batting_team.options[0]
        batting_team.set_value(first_team)
        bowling_team.set_value(first_team)
        submit_btn.click().run()

        error_texts = [e.value for e in self.at.error]
        self.assertTrue(
            any("different" in err.lower() for err in error_texts),
            f"Expected error for same team, got: {error_texts}",
        )
        self.assertEqual(len(self.at.success), 0, "Prediction generated despite validation error")

        # Sub-case B: Runs in last 5 overs exceeding total score
        bowling_team.set_value(batting_team.options[1])
        score_input.set_value(40)
        runs_last_5.set_value(50)
        submit_btn.click().run()

        error_texts = [e.value for e in self.at.error]
        self.assertTrue(
            any("cannot exceed" in err.lower() for err in error_texts),
            f"Expected error for runs_last_5 > score, got: {error_texts}",
        )

    def test_first_innings_widget_boundary_specifications(self):
        """Widget boundary parameters prevent out-of-range inputs at the UI level."""
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        wickets_input = [n for n in self.at.number_input if n.label == "Wickets Lost (0–10)"][0]
        runs_last_5 = [n for n in self.at.number_input if n.label == "Runs in Overs 5–9"][0]
        wickets_last_5 = [n for n in self.at.number_input if n.label == "Wickets in Overs 5–9"][0]

        self.assertEqual(score_input.min, 0.0)
        self.assertEqual(wickets_input.min, 0.0)
        self.assertEqual(wickets_input.max, 10.0)
        self.assertEqual(runs_last_5.min, 0.0)
        self.assertEqual(wickets_last_5.min, 0.0)
        self.assertEqual(wickets_last_5.max, 10.0)


class TestChaseWinProbabilitySection(unittest.TestCase):
    """Test scenario 4: Chase-Win Probability, situation metrics, and boundaries."""

    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.at.sidebar.radio[0].set_value("3. Chase-Win Probability").run()

    def test_chase_widgets_exist_with_valid_defaults(self):
        """All expected chase widgets and probabilistic classifiers are present."""
        selectbox_labels = [s.label for s in self.at.selectbox]
        self.assertIn("Chasing Team (Batting 2nd)", selectbox_labels)
        self.assertIn("Defending Team (Bowling 2nd)", selectbox_labels)
        self.assertIn("Venue / Stadium", selectbox_labels)
        self.assertIn("Probabilistic Classifier", selectbox_labels)

        number_labels = [n.label for n in self.at.number_input]
        self.assertIn("Target to Win", number_labels)
        self.assertIn("Score at 10 Overs", number_labels)
        self.assertIn("Wickets Lost (0–10)", number_labels)
        self.assertIn("Runs in Overs 5–9", number_labels)
        self.assertIn("Wickets in Overs 5–9", number_labels)

        button_labels = [b.label for b in self.at.button]
        self.assertIn("Estimate Chase-Win Probability", button_labels)

        # Verify Probabilistic Classifier choices
        chase_sb = [s for s in self.at.selectbox if s.label == "Probabilistic Classifier"][0]
        for expected_model in ["Random Forest", "Logistic Regression", "Gradient Boosting", "K-Nearest Neighbors"]:
            self.assertIn(expected_model, chase_sb.options)

    def test_valid_submission_produces_percentage_and_situation_metrics(self):
        """Valid chase inputs yield probability as percentage and key situation metrics."""
        submit_btn = [b for b in self.at.button if b.label == "Estimate Chase-Win Probability"][0]
        submit_btn.click().run()

        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(len(self.at.error), 0)

        # Check success message
        success_msgs = [s.value for s in self.at.success]
        self.assertTrue(any("Chase Situation & Win Probability" in m for m in success_msgs))

        metric_dict = {m.label: m.value for m in self.at.metric}
        self.assertIn("Win Probability (Chasing)", metric_dict)
        self.assertIn("Runs Needed", metric_dict)
        self.assertIn("Required Run Rate", metric_dict)
        self.assertIn("Current Run Rate", metric_dict)

        # Verify probability percentage format and bounds [0.0%, 100.0%]
        prob_str = metric_dict["Win Probability (Chasing)"]
        self.assertTrue(re.match(r"^\d+(\.\d+)?%$", prob_str), f"Invalid probability format: {prob_str}")
        prob_val = float(prob_str.replace("%", "")) / 100.0
        self.assertGreaterEqual(prob_val, 0.0)
        self.assertLessEqual(prob_val, 1.0)

        # Check progress bar presence and value
        self.assertGreaterEqual(len(self.at.progress), 1)
        prog_val = self.at.progress[0].value
        # Streamlit progress can be 0.0-1.0 or 0-100 depending on version
        normalized_prog = prog_val / 100.0 if prog_val > 1.0 else prog_val
        self.assertAlmostEqual(normalized_prog, prob_val, delta=0.02)

    def test_changing_chase_inputs_updates_win_probability(self):
        """Dominant vs difficult chase scenarios produce distinct, appropriate probabilities."""
        target_input = [n for n in self.at.number_input if n.label == "Target to Win"][0]
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        wickets_input = [n for n in self.at.number_input if n.label == "Wickets Lost (0–10)"][0]
        runs_last_5 = [n for n in self.at.number_input if n.label == "Runs in Overs 5–9"][0]
        wickets_last_5 = [n for n in self.at.number_input if n.label == "Wickets in Overs 5–9"][0]
        submit_btn = [b for b in self.at.button if b.label == "Estimate Chase-Win Probability"][0]

        # Case 1: Dominant chase (easy target, strong position at 10 overs)
        target_input.set_value(120)
        score_input.set_value(85)
        wickets_input.set_value(1)
        runs_last_5.set_value(45)
        wickets_last_5.set_value(0)
        submit_btn.click().run()

        metrics_1 = {m.label: m.value for m in self.at.metric}
        prob_1 = float(metrics_1["Win Probability (Chasing)"].replace("%", ""))

        # Case 2: Improbable chase (high target, collapsed wickets at 10 overs)
        target_input.set_value(220)
        score_input.set_value(65)
        wickets_input.set_value(6)
        runs_last_5.set_value(25)
        wickets_last_5.set_value(3)
        submit_btn.click().run()

        metrics_2 = {m.label: m.value for m in self.at.metric}
        prob_2 = float(metrics_2["Win Probability (Chasing)"].replace("%", ""))

        self.assertGreater(prob_1, prob_2, f"Expected dominant chase prob {prob_1}% > difficult {prob_2}%")

    def test_semantic_invalid_inputs_display_validation_errors(self):
        """Invalid chase inputs display error messages and prevent prediction."""
        chasing_team = [s for s in self.at.selectbox if s.label == "Chasing Team (Batting 2nd)"][0]
        defending_team = [s for s in self.at.selectbox if s.label == "Defending Team (Bowling 2nd)"][0]
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        runs_last_5 = [n for n in self.at.number_input if n.label == "Runs in Overs 5–9"][0]
        submit_btn = [b for b in self.at.button if b.label == "Estimate Chase-Win Probability"][0]

        # Sub-case A: Same team for chase and defense
        team_name = chasing_team.options[0]
        chasing_team.set_value(team_name)
        defending_team.set_value(team_name)
        submit_btn.click().run()

        error_texts = [e.value for e in self.at.error]
        self.assertTrue(
            any("different" in err.lower() for err in error_texts),
            f"Expected error for same team, got: {error_texts}",
        )
        self.assertEqual(len(self.at.success), 0)

        # Sub-case B: Runs in last 5 overs exceeding score
        defending_team.set_value(chasing_team.options[1])
        score_input.set_value(50)
        runs_last_5.set_value(60)
        submit_btn.click().run()

        error_texts = [e.value for e in self.at.error]
        self.assertTrue(
            any("cannot exceed" in err.lower() for err in error_texts),
            f"Expected error for runs_last_5 > score, got: {error_texts}",
        )

    def test_chase_widget_boundary_specifications(self):
        """Boundary constraints on chase number inputs prevent invalid ranges at the UI layer."""
        target_input = [n for n in self.at.number_input if n.label == "Target to Win"][0]
        score_input = [n for n in self.at.number_input if n.label == "Score at 10 Overs"][0]
        wickets_input = [n for n in self.at.number_input if n.label == "Wickets Lost (0–10)"][0]

        # Target must be strictly positive (min >= 1)
        self.assertGreaterEqual(target_input.min, 1.0)
        # Score must be non-negative (min >= 0)
        self.assertEqual(score_input.min, 0.0)
        # Wickets must be between 0 and 10
        self.assertEqual(wickets_input.min, 0.0)
        self.assertEqual(wickets_input.max, 10.0)


class TestModelEvaluationSection(unittest.TestCase):
    """Test scenario 5: Model Evaluation tables, metric integrity, and error handling."""

    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.at.sidebar.radio[0].set_value("4. Model Evaluation").run()

    def test_evaluation_tabs_and_headings(self):
        """Header, tabs, and subheaders exist for all three model tasks."""
        headers = [h.value for h in self.at.header]
        self.assertIn("Held-Out Test Set Model Evaluation", headers)

        tab_labels = [t.label for t in self.at.tabs]
        self.assertIn("1. First-Innings Regression", tab_labels)
        self.assertIn("2. First-Innings Classification", tab_labels)
        self.assertIn("3. Chase-Win Probability", tab_labels)

        subheaders = [sh.value for sh in self.at.subheader]
        self.assertIn("First-Innings Regression (MAE & R²)", subheaders)
        self.assertIn("First-Innings Score Classification", subheaders)
        self.assertIn("Chase-Win Probability Prediction", subheaders)

    def test_metrics_sourced_from_existing_evaluation_code(self):
        """DataFrames match the live evaluation data produced by existing modules."""
        data_bundle = load_all_models_and_metrics()

        # There are 3 dataframes: Regression, Classification, Chase
        self.assertEqual(len(self.at.dataframe), 3)

        # Tab 1: Regression DataFrame
        reg_df = self.at.dataframe[0].value
        self.assertIn("Model", reg_df.columns)
        self.assertIn("Test MAE (Runs)", reg_df.columns)
        self.assertIn("Test R²", reg_df.columns)

        expected_reg_metrics = data_bundle["regression"]["metrics"]
        self.assertEqual(len(reg_df), len(expected_reg_metrics))
        for _, row in reg_df.iterrows():
            model_name = row["Model"]
            self.assertIn(model_name, expected_reg_metrics)
            self.assertAlmostEqual(row["Test MAE (Runs)"], expected_reg_metrics[model_name]["mae"], places=3)
            self.assertAlmostEqual(row["Test R²"], expected_reg_metrics[model_name]["r2"], places=3)

        # Tab 2: Classification DataFrame
        clf_df = self.at.dataframe[1].value
        self.assertIn("Model", clf_df.columns)
        self.assertIn("Accuracy", clf_df.columns)
        self.assertIn("Macro F1", clf_df.columns)

        expected_clf_models = data_bundle["classification"]["models"]
        self.assertEqual(len(clf_df), len(expected_clf_models))
        for _, row in clf_df.iterrows():
            model_name = row["Model"]
            self.assertIn(model_name, expected_clf_models)
            self.assertAlmostEqual(row["Accuracy"], expected_clf_models[model_name]["accuracy"], places=3)
            self.assertAlmostEqual(row["Macro F1"], expected_clf_models[model_name]["macro_f1"], places=3)

        # Tab 3: Chase-Win DataFrame
        chase_df = self.at.dataframe[2].value
        self.assertIn("Model", chase_df.columns)
        self.assertIn("Accuracy", chase_df.columns)
        self.assertIn("ROC-AUC", chase_df.columns)
        self.assertIn("Brier Score", chase_df.columns)

        expected_chase_models = data_bundle["chase"]["models"]
        self.assertEqual(len(chase_df), len(expected_chase_models))
        for _, row in chase_df.iterrows():
            model_name = row["Model"]
            self.assertIn(model_name, expected_chase_models)
            self.assertAlmostEqual(row["Accuracy"], expected_chase_models[model_name]["accuracy"], places=3)
            self.assertAlmostEqual(row["ROC-AUC"], expected_chase_models[model_name]["roc_auc"], places=3)
            self.assertAlmostEqual(row["Brier Score"], expected_chase_models[model_name]["brier_score"], places=3)

    def test_missing_metrics_handled_clearly(self):
        """Missing or failed evaluation metrics display clear warnings rather than crashing."""
        st.cache_resource.clear()
        try:
            # Test empty metrics handling
            empty_bundle = {"regression": {}, "classification": {}, "chase": {}}
            with patch("src.dashboard_utils.load_all_models_and_metrics", return_value=empty_bundle):
                at_empty = AppTest.from_file(APP_PATH, default_timeout=30).run()
                at_empty.sidebar.radio[0].set_value("4. Model Evaluation").run()

                self.assertEqual(len(at_empty.exception), 0)
                warning_texts = [w.value for w in at_empty.warning]
                self.assertTrue(any("regression" in w.lower() for w in warning_texts))
                self.assertTrue(any("classification" in w.lower() for w in warning_texts))
                self.assertTrue(any("chase" in w.lower() for w in warning_texts))

            # Test exception in loading metrics
            st.cache_resource.clear()
            with patch("src.dashboard_utils.load_all_models_and_metrics", side_effect=RuntimeError("Data load error")):
                at_err = AppTest.from_file(APP_PATH, default_timeout=30).run()
                at_err.sidebar.radio[0].set_value("4. Model Evaluation").run()

                self.assertEqual(len(at_err.exception), 0)
                error_texts = [e.value for e in at_err.error]
                self.assertTrue(any("Failed to load" in e for e in error_texts))
        finally:
            st.cache_resource.clear()


class TestRegressionAndRepeatability(unittest.TestCase):
    """Test scenario 6: Rerunning, caching behavior, and execution repeatability."""

    def test_rerunning_app_does_not_cause_exceptions(self):
        """Repeated sequential runs do not raise unhandled exceptions or leak state."""
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        for _ in range(3):
            at.run()
            self.assertEqual(len(at.exception), 0)

    def test_cached_data_and_models_consistent_across_runs(self):
        """Cached dataset overview and models return identical results without mutation."""
        data_1 = get_project_overview_data()
        data_2 = get_project_overview_data()
        self.assertEqual(data_1, data_2)

        models_1 = load_all_models_and_metrics()
        models_2 = load_all_models_and_metrics()
        self.assertEqual(
            list(models_1["regression"]["metrics"].keys()),
            list(models_2["regression"]["metrics"].keys()),
        )

    def test_order_independence_of_navigation(self):
        """Sections can be loaded in arbitrary non-sequential order without cross-test leakage."""
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        arbitrary_order = [
            "3. Chase-Win Probability",
            "1. Project Overview",
            "5. Batter Performance Clustering",
            "4. Model Evaluation",
            "2. Final Score Prediction & Class",
        ]
        for sec in arbitrary_order:
            at.sidebar.radio[0].set_value(sec).run()
            self.assertEqual(
                len(at.exception),
                0,
                f"Exception encountered visiting '{sec}' in non-sequential order: {at.exception}",
            )


class TestBatterClusteringSection(unittest.TestCase):
    """Test scenario: Batter Performance Clustering layout, metrics, tables, and error handling."""

    def setUp(self):
        self.at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.at.sidebar.radio[0].set_value("5. Batter Performance Clustering").run()

    def test_clustering_section_renders_cleanly(self):
        """Batter Performance Clustering section renders without uncaught exceptions."""
        self.assertEqual(len(self.at.exception), 0, f"Uncaught exceptions in clustering section: {self.at.exception}")
        headers = [h.value for h in self.at.header]
        self.assertIn("Batter Performance Clustering", headers)

    def test_clustering_metrics_and_summary_tables(self):
        """Clustering summary metrics, candidate k scores, and tables are properly populated."""
        from src.batter_clustering import run_batter_clustering
        expected = run_batter_clustering()

        metric_dict = {m.label: m.value for m in self.at.metric}
        self.assertIn("Eligible Batters Evaluated", metric_dict)
        self.assertIn("Optimal Clusters (k)", metric_dict)
        self.assertIn("Selected Silhouette Score", metric_dict)
        self.assertEqual(metric_dict["Eligible Batters Evaluated"], str(expected["total_batters_evaluated"]))
        self.assertEqual(metric_dict["Optimal Clusters (k)"], str(expected["chosen_k"]))

        # Dataframes: candidate scores table, cluster summary table, assigned batters table
        self.assertGreaterEqual(len(self.at.dataframe), 3)

    def test_clustering_error_handling_graceful(self):
        """Clustering failures display clear error message without crashing the dashboard."""
        st.cache_resource.clear()
        try:
            with patch("src.batter_clustering.run_batter_clustering", side_effect=RuntimeError("Clustering test failure")):
                at_err = AppTest.from_file(APP_PATH, default_timeout=30).run()
                at_err.sidebar.radio[0].set_value("5. Batter Performance Clustering").run()

                self.assertEqual(len(at_err.exception), 0)
                errors = [e.value for e in at_err.error]
                self.assertTrue(any("Clustering test failure" in e or "Error" in e for e in errors))
        finally:
            st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
