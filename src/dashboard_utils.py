"""
Dashboard utilities and validation helpers for the IPL Match Intelligence Lab.

Provides reusable inference helpers, input validation, and metric retrieval
for the Streamlit dashboard without duplicating ML or preprocessing logic.
"""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.chase import (
    CHASE_CATEGORICAL_FEATURES,
    CHASE_NUMERIC_FEATURES,
    run_chase_win_prediction,
)
from src.classification import (
    assign_score_classes,
    run_first_innings_classification,
)
from src.preprocessing import load_and_preprocess_data, normalize_venue
from src.regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    run_first_innings_regression,
)
from src.snapshots import create_ten_over_snapshots


def validate_first_innings_inputs(
    current_score: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
    batting_team: str,
    bowling_team: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate user inputs for first-innings score prediction:
    - Current score must be non-negative.
    - Wickets lost must be in range [0, 10].
    - Runs in last 5 overs must be non-negative and not exceed current score.
    - Wickets in last 5 overs must be non-negative and not exceed total wickets.
    - Batting and bowling teams must be distinct.
    """
    if batting_team == bowling_team:
        return False, "Batting team and Bowling team must be different."
    if current_score < 0:
        return False, "Current score cannot be negative."
    if not (0 <= wickets <= 10):
        return False, f"Wickets lost must be between 0 and 10 (got {wickets})."
    if runs_last_5 < 0:
        return False, "Runs scored in last 5 overs cannot be negative."
    if runs_last_5 > current_score:
        return False, f"Runs in last 5 overs ({runs_last_5}) cannot exceed total score ({current_score})."
    if wickets_last_5 < 0:
        return False, "Wickets lost in last 5 overs cannot be negative."
    if wickets_last_5 > wickets:
        return False, f"Wickets in last 5 overs ({wickets_last_5}) cannot exceed total wickets ({wickets})."

    return True, None


def validate_chase_inputs(
    target: int,
    current_score: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
    batting_team: str,
    bowling_team: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate user inputs for second-innings chase prediction:
    - Target must be strictly positive.
    - Current score must be non-negative.
    - Wickets lost must be in range [0, 10].
    - Runs in last 5 overs must be non-negative and not exceed current score.
    - Wickets in last 5 overs must be non-negative and not exceed total wickets.
    - Batting and bowling teams must be distinct.
    """
    if batting_team == bowling_team:
        return False, "Chasing team and Defending team must be different."
    if target <= 0:
        return False, "Target score must be a positive integer greater than 0."
    if current_score < 0:
        return False, "Current score cannot be negative."
    if not (0 <= wickets <= 10):
        return False, f"Wickets lost must be between 0 and 10 (got {wickets})."
    if runs_last_5 < 0:
        return False, "Runs in last 5 overs cannot be negative."
    if runs_last_5 > current_score:
        return False, f"Runs in last 5 overs ({runs_last_5}) cannot exceed total score ({current_score})."
    if wickets_last_5 < 0:
        return False, "Wickets in last 5 overs cannot be negative."
    if wickets_last_5 > wickets:
        return False, f"Wickets in last 5 overs ({wickets_last_5}) cannot exceed total wickets ({wickets})."

    return True, None


def build_first_innings_input_df(
    batting_team: str,
    bowling_team: str,
    venue: str,
    current_score: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
) -> pd.DataFrame:
    """
    Construct a single-row feature DataFrame formatted for the first-innings regression
    and classification pipelines.
    """
    row = {
        "current_score": int(current_score),
        "wickets": int(wickets),
        "runs_last_5": int(runs_last_5),
        "wickets_last_5": int(wickets_last_5),
        "batting_team": str(batting_team),
        "bowling_team": str(bowling_team),
        "venue": normalize_venue(str(venue)) if venue is not None else None,
    }
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame([row])[cols]


def build_chase_input_df(
    batting_team: str,
    bowling_team: str,
    venue: str,
    target: int,
    current_score: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
) -> pd.DataFrame:
    """
    Construct a single-row feature DataFrame for the second-innings chase prediction
    pipeline, calculating derived features:
    - runs_needed = target - current_score
    - required_run_rate = runs_needed / 10.0
    - current_run_rate = current_score / 10.0
    """
    target_int = int(target)
    score_int = int(current_score)
    runs_needed = target_int - score_int

    row = {
        "target": target_int,
        "current_score": score_int,
        "wickets": int(wickets),
        "runs_last_5": int(runs_last_5),
        "wickets_last_5": int(wickets_last_5),
        "runs_needed": runs_needed,
        "required_run_rate": round(runs_needed / 10.0, 2),
        "current_run_rate": round(score_int / 10.0, 2),
        "batting_team": str(batting_team),
        "bowling_team": str(bowling_team),
        "venue": normalize_venue(str(venue)) if venue is not None else None,
    }
    cols = CHASE_NUMERIC_FEATURES + CHASE_CATEGORICAL_FEATURES
    return pd.DataFrame([row])[cols]


def get_teams_and_venues(
    matches_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], List[str]]:
    """Retrieve sorted lists of all standardized teams and venues."""
    if matches_df is None:
        matches_df, _, _ = load_and_preprocess_data()

    teams = sorted(list(set(matches_df["team1"].dropna()) | set(matches_df["team2"].dropna())))
    venues = sorted(list(set(matches_df["venue"].dropna().apply(normalize_venue))))

    return teams, venues


def get_project_overview_data() -> Dict[str, Any]:
    """
    Compute actual dataset sizes and snapshot counts directly from raw and processed data.
    """
    matches_df, deliveries_df, audit = load_and_preprocess_data()
    snap1, snap2 = create_ten_over_snapshots(matches_df, deliveries_df)

    snap1_train = snap1[snap1["season"] <= 2021]
    snap1_test = snap1[snap1["season"] >= 2022]
    snap2_train = snap2[snap2["season"] <= 2021]
    snap2_test = snap2[snap2["season"] >= 2022]

    return {
        "raw_matches_count": audit["rows_matches_before"],
        "raw_deliveries_count": audit["rows_deliveries_before"],
        "cleaned_matches_count": audit["rows_matches_after"],
        "first_innings_total": len(snap1),
        "first_innings_train": len(snap1_train),
        "first_innings_test": len(snap1_test),
        "second_innings_total": len(snap2),
        "second_innings_train": len(snap2_train),
        "second_innings_test": len(snap2_test),
        "train_seasons_range": "2008–2021",
        "test_seasons_range": "2022–2024",
    }


def load_all_models_and_metrics() -> Dict[str, Any]:
    """
    Load trained pipelines and verified evaluation metrics across Phases 1, 2, and 3.
    Reuses existing module functions without retraining or inventing metrics.
    """
    reg_summary = run_first_innings_regression()
    clf_summary = run_first_innings_classification()
    chase_summary = run_chase_win_prediction()

    return {
        "regression": reg_summary,
        "classification": clf_summary,
        "chase": chase_summary,
    }
