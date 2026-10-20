"""
Snapshot extraction module for the IPL Match Intelligence Lab.

Constructs 10-over match snapshots (overs 0-9) for both first-innings
score prediction and second-innings chase analysis.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd


def count_wickets_lost(deliveries_df: pd.DataFrame) -> int:
    """
    Calculate the number of wickets lost by the batting team in a set of deliveries.

    Cricket dismissal rules:
    - Retired Hurt: Under Law 25, a batsman who retires hurt or ill is 'retired not out'.
      This does NOT count as a lost wicket for the batting team.
    - Run Out: A run out is a dismissal where the team loses a wicket. It counts toward
      wickets lost by the team, even though it is not credited to the bowler.
    - All other dismissals (bowled, caught, caught and bowled, lbw, stumped, hit wicket,
      obstructing the field, retired out) are standard dismissals and count as lost wickets.

    Rule: is_wicket == 1 AND dismissal_kind != 'retired hurt'
    """
    if deliveries_df.empty:
        return 0
    is_w = deliveries_df["is_wicket"] == 1
    not_retired_hurt = deliveries_df["dismissal_kind"] != "retired hurt"
    return int((is_w & not_retired_hurt).sum())


def has_completed_10_overs(innings_deliveries: pd.DataFrame) -> bool:
    """
    Determine whether an innings has at least 10 completed overs (over indices 0-9).

    In T20 cricket, each over consists of 6 legal balls. Over indexing in the dataset is 0 to 19.
    An innings has completed 10 overs if:
    1. Overs 0 through 9 are all present in the innings deliveries.
    2. The innings either:
       a) Proceeded into over 10 or later (over >= 10), confirming over 9 was completed, OR
       b) Over 9 finished all 6 legal deliveries (excluding illegal deliveries: wides and noballs).
    Innings ending during over index 9 without completing 6 legal balls return False.
    """
    if innings_deliveries.empty:
        return False

    overs_present = set(innings_deliveries["over"].unique())
    if not set(range(10)).issubset(overs_present):
        return False

    max_over = innings_deliveries["over"].max()
    if max_over >= 10:
        return True
    if max_over == 9:
        over_9 = innings_deliveries[innings_deliveries["over"] == 9]
        # In cricket, wides and no-balls are illegal deliveries and must be re-bowled
        legal_balls = over_9[~over_9["extras_type"].isin(["wides", "noballs"])]
        return len(legal_balls) >= 6
    return False


# Explicit separation of input features from target labels to prevent data leakage in modeling
FIRST_INNINGS_FEATURES: List[str] = [
    "match_id",
    "season",
    "venue",
    "city",
    "batting_team",
    "bowling_team",
    "current_score",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
]
FIRST_INNINGS_TARGET: str = "final_score"

TOSS_FEATURES: List[str] = [
    "toss_winner",
    "toss_decision",
]

SECOND_INNINGS_FEATURES: List[str] = [
    "match_id",
    "season",
    "venue",
    "city",
    "batting_team",
    "bowling_team",
    "target",
    "current_score",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
    "runs_needed",
    "balls_remaining",
    "required_run_rate",
    "current_run_rate",
]
SECOND_INNINGS_TARGET: str = "target_achieved"


def is_eligible_match(match_row: pd.Series) -> bool:
    """
    Check if a match is suitable for ordinary 20-over modeling:
    - Excludes Duckworth-Lewis (D/L) matches (method is not null)
    - Excludes super-over matches (super_over == 'Y')
    - Excludes abandoned matches (result == 'no result')
    - Excludes rain-reduced matches (target_overs must equal 20)
    """
    if pd.notna(match_row.get("method")):
        return False
    if match_row.get("super_over") == "Y":
        return False
    if match_row.get("result") == "no result":
        return False
    if match_row.get("target_overs") != 20:
        return False
    return True


def create_first_innings_snapshots(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create 10-over snapshots for first innings of eligible matches.

    Snapshots capture match state at exactly 10 completed overs (indices 0 to 9):
    - match_id, season, venue, city
    - batting_team, bowling_team
    - current_score (total runs including extras in overs 0-9)
    - wickets (wickets lost in overs 0-9, excluding retired hurt)
    - runs_last_5 (runs scored in overs 5-9)
    - wickets_last_5 (wickets lost in overs 5-9)
    - final_score (total 1st innings runs across all 20 overs, target for prediction)
    """
    grouped_deliv = {k: v for k, v in deliveries_df.groupby(["match_id", "inning"])}
    records: List[Dict] = []

    for _, match in matches_df.iterrows():
        if not is_eligible_match(match):
            continue

        mid = match["id"]
        key = (mid, 1)
        if key not in grouped_deliv:
            continue

        d1 = grouped_deliv[key]
        if not has_completed_10_overs(d1):
            continue

        # Overs 0 to 9 (first 10 completed overs)
        d1_10 = d1[d1["over"] <= 9]
        # Overs 5 to 9 (most recent 5 completed overs)
        d1_last5 = d1[d1["over"].isin([5, 6, 7, 8, 9])]

        records.append({
            "match_id": mid,
            "season": match["season"],
            "venue": match.get("venue"),
            "city": match.get("city"),
            "toss_winner": match.get("toss_winner"),
            "toss_decision": match.get("toss_decision"),
            "batting_team": d1["batting_team"].iloc[0],
            "bowling_team": d1["bowling_team"].iloc[0],
            "current_score": int(d1_10["total_runs"].sum()),
            "wickets": count_wickets_lost(d1_10),
            "runs_last_5": int(d1_last5["total_runs"].sum()),
            "wickets_last_5": count_wickets_lost(d1_last5),
            "final_score": int(d1["total_runs"].sum()),
        })

    return pd.DataFrame(records)


def create_second_innings_snapshots(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create 10-over snapshots for second innings (run chase) of eligible matches.

    Snapshots capture match state at exactly 10 completed overs (indices 0 to 9)
    only when a valid target and a completed 10-over chase observation exist:
    - match_id, season, venue, city
    - batting_team (chasing team), bowling_team (defending team)
    - target (target runs required to win)
    - current_score (chasing team runs in overs 0-9)
    - wickets (chasing team wickets lost in overs 0-9)
    - runs_last_5 (runs scored in overs 5-9)
    - wickets_last_5 (wickets lost in overs 5-9)
    - runs_needed (target - current_score)
    - balls_remaining (60 balls remaining at the 10-over mark of a 20-over match)
    - required_run_rate ((target - current_score) / 10.0 overs)
    - current_run_rate (current_score / 10.0 overs)
    - winner (actual match winner)
    - target_achieved (1 if batting_team won the chase, 0 otherwise)
    """
    grouped_deliv = {k: v for k, v in deliveries_df.groupby(["match_id", "inning"])}
    records: List[Dict] = []

    for _, match in matches_df.iterrows():
        if not is_eligible_match(match):
            continue

        target_runs = match.get("target_runs")
        if pd.isna(target_runs) or target_runs <= 0:
            continue

        mid = match["id"]
        key = (mid, 2)
        if key not in grouped_deliv:
            continue

        d2 = grouped_deliv[key]
        if not has_completed_10_overs(d2):
            continue

        # Overs 0 to 9 (first 10 completed overs of chase)
        d2_10 = d2[d2["over"] <= 9]
        # Overs 5 to 9 (most recent 5 completed overs of chase)
        d2_last5 = d2[d2["over"].isin([5, 6, 7, 8, 9])]

        target = int(target_runs)
        current_score = int(d2_10["total_runs"].sum())
        batting_team = d2["batting_team"].iloc[0]
        winner = match.get("winner")

        records.append({
            "match_id": mid,
            "season": match["season"],
            "venue": match.get("venue"),
            "city": match.get("city"),
            "toss_winner": match.get("toss_winner"),
            "toss_decision": match.get("toss_decision"),
            "batting_team": batting_team,
            "bowling_team": d2["bowling_team"].iloc[0],
            "target": target,
            "current_score": current_score,
            "wickets": count_wickets_lost(d2_10),
            "runs_last_5": int(d2_last5["total_runs"].sum()),
            "wickets_last_5": count_wickets_lost(d2_last5),
            "runs_needed": target - current_score,
            "balls_remaining": 60,
            "required_run_rate": round((target - current_score) / 10.0, 2),
            "current_run_rate": round(current_score / 10.0, 2),
            "winner": winner,
            "target_achieved": 1 if winner == batting_team else 0,
        })

    return pd.DataFrame(records)


def create_ten_over_snapshots(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate both first-innings and second-innings
    10-over snapshots.
    Returns: (first_innings_snapshots_df, second_innings_snapshots_df)
    """
    df1 = create_first_innings_snapshots(matches_df, deliveries_df)
    df2 = create_second_innings_snapshots(matches_df, deliveries_df)
    return df1, df2
