"""
Stadium-Specific Par Score and Classification Module for IPL Match Intelligence Lab.

Computes historical first-innings stadium scoring statistics (par scores,
distributions, and percentiles) and classifies regression-predicted scores into
Low, Par, or High relative to the specific venue's conditions without using an ML classifier.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.preprocessing import (
    load_and_preprocess_data,
    normalize_venue,
    resolve_data_path,
)
from src.snapshots import create_first_innings_snapshots, is_eligible_match

DEFAULT_MATCHES_PATH: str = "data/matches_2008-2024.csv"
DEFAULT_DELIVERIES_PATH: str = "data/deliveries_2008-2024.csv"
DEFAULT_PAR_FILE: str = "data/stadium_par_scores.csv"
FALLBACK_VENUE_KEY: str = "_COMPETITION_AVERAGE_"


def compute_stadium_par_scores(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    max_season: Optional[int] = 2021,
    before_date: Optional[str] = None,
    min_matches_threshold: int = 5,
    shrinkage_m: float = 5.0,
    recent_seasons_window: int = 4,
    lower_quantile: float = 1.0 / 3.0,
    upper_quantile: float = 2.0 / 3.0,
) -> pd.DataFrame:
    """
    Calculate historical first-innings scoring statistics for each stadium strictly
    before the cutoff point (max_season / before_date) to prevent temporal data leakage.

    Parameters:
    - matches_df: Standardized matches DataFrame.
    - deliveries_df: Standardized deliveries DataFrame.
    - max_season: Strict chronological season cutoff (default 2021, matching train set).
    - before_date: Optional ISO date string (YYYY-MM-DD) for rolling point-in-time calculation.
    - min_matches_threshold: Matches below which shrinkage heavily weights competition average.
    - shrinkage_m: Empirical Bayes pseudo-count weighting for sample shrinkage.
    - recent_seasons_window: Number of recent seasons to evaluate short-term scoring trends.
    - lower_quantile: Low boundary tertile (default 33.33%).
    - upper_quantile: High boundary tertile (default 66.67%).

    Returns:
    - pd.DataFrame containing historical stadium statistics and a competition average fallback row.
    """
    # 1. Use verified first-innings snapshots of eligible 20-over matches
    snapshots = create_first_innings_snapshots(matches_df, deliveries_df)

    # 2. Strict chronological filtering to prevent test-set or future leakage
    if max_season is not None:
        snapshots = snapshots[snapshots["season"] <= max_season].copy()

    if before_date is not None and "date" in matches_df.columns:
        match_dates = matches_df.set_index("id")["date"].to_dict()
        snapshots["match_date"] = snapshots["match_id"].map(match_dates)
        snapshots = snapshots[snapshots["match_date"] < before_date].copy()

    if snapshots.empty:
        raise ValueError("No eligible first-innings match snapshots found within the specified time window.")

    # 3. Global competition baseline across all eligible training matches
    global_scores = snapshots["final_score"].astype(float)
    g_count = int(len(global_scores))
    g_mean = float(global_scores.mean())
    g_median = float(global_scores.median())
    g_std = float(global_scores.std()) if g_count > 1 else 0.0
    g_p33 = float(global_scores.quantile(lower_quantile))
    g_p66 = float(global_scores.quantile(upper_quantile))

    # Recent scoring trend baseline (e.g., last 4 seasons of training period)
    max_train_season = int(snapshots["season"].max())
    recent_season_cutoff = max_train_season - recent_seasons_window + 1
    recent_global = snapshots[snapshots["season"] >= recent_season_cutoff]["final_score"].astype(float)
    g_recent_mean = float(recent_global.mean()) if not recent_global.empty else g_mean

    rows: List[Dict[str, Any]] = []

    # 4. Calculate per-venue statistics
    for raw_venue, group in snapshots.groupby("venue"):
        if pd.isna(raw_venue):
            continue
        venue = normalize_venue(str(raw_venue))
        scores = group["final_score"].astype(float)
        n = int(len(scores))

        v_mean = float(scores.mean())
        v_median = float(scores.median())
        v_std = float(scores.std()) if n > 1 else 0.0
        v_p33 = float(scores.quantile(lower_quantile))
        v_p66 = float(scores.quantile(upper_quantile))

        # Recent trend at this specific venue
        recent_scores = group[group["season"] >= recent_season_cutoff]["final_score"].astype(float)
        v_recent_mean = float(recent_scores.mean()) if not recent_scores.empty else v_mean

        # Empirical Bayesian shrinkage to avoid volatility on small samples
        # weight w -> 1 as n becomes large; weight w -> 0 as n -> 0
        w = n / (n + shrinkage_m)

        shrunk_par = round(float(w * v_median + (1.0 - w) * g_median), 1)
        shrunk_low = round(float(w * v_p33 + (1.0 - w) * g_p33), 1)
        shrunk_high = round(float(w * v_p66 + (1.0 - w) * g_p66), 1)

        # Integrity safeguard: guarantee low threshold is strictly less than high threshold
        if shrunk_low >= shrunk_high:
            shrunk_low = round(shrunk_par - 10.0, 1)
            shrunk_high = round(shrunk_par + 10.0, 1)

        rows.append({
            "venue": venue,
            "match_count": n,
            "median_score": round(v_median, 1),
            "mean_score": round(v_mean, 1),
            "std_score": round(v_std, 1),
            "p33_score": round(v_p33, 1),
            "p66_score": round(v_p66, 1),
            "recent_mean_score": round(v_recent_mean, 1),
            "par_score": shrunk_par,
            "threshold_low": shrunk_low,
            "threshold_high": shrunk_high,
        })

    # 5. Add competition fallback row for unseen or missing stadiums
    rows.append({
        "venue": FALLBACK_VENUE_KEY,
        "match_count": g_count,
        "median_score": round(g_median, 1),
        "mean_score": round(g_mean, 1),
        "std_score": round(g_std, 1),
        "p33_score": round(g_p33, 1),
        "p66_score": round(g_p66, 1),
        "recent_mean_score": round(g_recent_mean, 1),
        "par_score": round(g_median, 1),
        "threshold_low": round(g_p33, 1),
        "threshold_high": round(g_p66, 1),
    })

    result_df = pd.DataFrame(rows)
    # Deduplicate venues if any aliasing produced duplicate rows by averaging
    if result_df["venue"].duplicated().any():
        result_df = result_df.groupby("venue", as_index=False).mean()

    return result_df.sort_values(by="match_count", ascending=False).reset_index(drop=True)


def save_stadium_par_scores(
    stats_df: pd.DataFrame,
    output_path: str = DEFAULT_PAR_FILE,
) -> str:
    """Save calculated stadium statistics to a CSV artifact."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_file, index=False)

    # Mirror to Datasets/ if it exists in the workspace
    datasets_dir = Path("Datasets")
    if datasets_dir.exists() and "data" in str(out_file):
        mirror_file = datasets_dir / out_file.name
        stats_df.to_csv(mirror_file, index=False)

    return str(out_file)


# Global in-memory cache to avoid reading from disk on repeated calls
_CACHED_PAR_STATS: Optional[pd.DataFrame] = None


def load_stadium_par_scores(path: str = DEFAULT_PAR_FILE) -> pd.DataFrame:
    """
    Load stadium statistics artifact from disk with caching and path resolution.
    Does not recalculate from raw ball-by-ball deliveries.
    """
    global _CACHED_PAR_STATS
    if _CACHED_PAR_STATS is not None:
        return _CACHED_PAR_STATS

    resolved = resolve_data_path(path, path.replace("data/", "Datasets/"))
    df = pd.read_csv(resolved)
    _CACHED_PAR_STATS = df
    return df


def classify_stadium_score(
    predicted_score: Union[int, float],
    venue: Optional[str] = None,
    stats_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Classify a continuous regression-predicted score as 'Low', 'Par', or 'High'
    relative to the historical scoring distribution of the specific stadium.

    Conditional Logic:
    - 'Low': predicted_score < threshold_low (tertile 1/3 bound)
    - 'Par': threshold_low <= predicted_score <= threshold_high
    - 'High': predicted_score > threshold_high (tertile 2/3 bound)

    Statistical Meaning:
    Describes score rarity relative to venue conditions. A 'High' classification
    indicates an above-average total historically for that ground, while 'Par'
    represents a typical competitive total. It is not an absolute win guarantee.
    """
    if stats_df is None:
        stats_df = load_stadium_par_scores()

    pred_val = float(predicted_score)
    norm_venue = normalize_venue(venue) if venue is not None else None

    # Retrieve venue row or fall back gracefully to competition average
    matched_row: Optional[pd.Series] = None
    is_fallback = False

    if norm_venue and norm_venue in stats_df["venue"].values:
        matched_row = stats_df[stats_df["venue"] == norm_venue].iloc[0]
    else:
        # Graceful fallback for new, rare, or missing venues
        matched_row = stats_df[stats_df["venue"] == FALLBACK_VENUE_KEY].iloc[0]
        is_fallback = True

    par_score = float(matched_row["par_score"])
    t_low = float(matched_row["threshold_low"])
    t_high = float(matched_row["threshold_high"])
    diff_from_par = round(pred_val - par_score, 1)

    if pred_val < t_low:
        classification = "Low"
    elif pred_val <= t_high:
        classification = "Par"
    else:
        classification = "High"

    return {
        "predicted_score": round(pred_val, 1),
        "stadium_par_score": round(par_score, 1),
        "difference_from_par": diff_from_par,
        "classification": classification,
        "venue": norm_venue if not is_fallback else "Competition Average",
        "threshold_low": round(t_low, 1),
        "threshold_high": round(t_high, 1),
        "sample_size": int(matched_row["match_count"]),
        "is_fallback": is_fallback,
    }


def predict_score_with_stadium_par(
    pipeline: Any,
    batting_team: str,
    bowling_team: str,
    venue: Optional[str],
    current_score: int,
    wickets: int,
    runs_last_5: int,
    wickets_last_5: int,
    stats_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    End-to-end integration helper:
    1. Runs the existing Regression pipeline without changing its features or weights.
    2. Retrieves saved stadium par statistics.
    3. Evaluates predicted score against stadium-specific boundaries.
    4. Returns exact structured response.
    """
    from src.regression import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    norm_venue = normalize_venue(str(venue)) if venue is not None else None
    row = {
        "current_score": int(current_score),
        "wickets": int(wickets),
        "runs_last_5": int(runs_last_5),
        "wickets_last_5": int(wickets_last_5),
        "batting_team": str(batting_team),
        "bowling_team": str(bowling_team),
        "venue": norm_venue,
    }
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = pd.DataFrame([row])[cols]

    pred_score = float(pipeline.predict(input_df)[0])
    return classify_stadium_score(pred_score, venue=venue, stats_df=stats_df)


def build_and_save_stadium_par_scores(
    matches_path: str = DEFAULT_MATCHES_PATH,
    deliveries_path: str = DEFAULT_DELIVERIES_PATH,
    output_path: str = DEFAULT_PAR_FILE,
    max_season: int = 2021,
) -> str:
    """
    Generate and save the stadium par scores artifact strictly using historical data
    up to max_season (2021) to prevent temporal leakage into held-out test seasons.
    """
    matches_df, deliveries_df, _ = load_and_preprocess_data(matches_path, deliveries_path)
    df = compute_stadium_par_scores(matches_df, deliveries_df, max_season=max_season)
    return save_stadium_par_scores(df, output_path=output_path)


if __name__ == "__main__":
    out = build_and_save_stadium_par_scores()
    print(f"Successfully generated and saved stadium par scores artifact to {out}")
