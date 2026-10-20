"""
IPL Batter Performance Clustering module for the IPL Match Intelligence Lab.

Unsupervised machine learning module using K-Means clustering to identify
distinct T20 batting profiles (e.g., high-impact anchors, aggressive finishers,
steady accumulators, bowling all-rounders / lower-order batters) based on
historical ball-by-ball delivery data.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean_deliveries, load_and_preprocess_data

# Chronological split season consistent with regression, classification, and chase modules
DEFAULT_SPLIT_SEASON: int = 2021

# Minimum balls faced threshold to filter out small-sample noise
# Batters with fewer than 200 balls faced lack statistical reliability for rate metrics
DEFAULT_MIN_BALLS: int = 200

# Primary clustering features: batting efficiency and scoring rate
DEFAULT_FEATURES: List[str] = ["batting_average", "strike_rate"]

# Hyperparameters for deterministic, reproducible K-Means clustering
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_N_INIT: int = 10
DEFAULT_K_RANGE: Tuple[int, int] = (2, 5)


def calculate_batter_statistics(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delivery-level records into comprehensive batter-level statistics.

    Cricket accounting rules applied:
    1. Total Runs: Sum of `batsman_runs` scored off the bat by the batter.
       Extras (wides, noballs, byes, legbyes, penalty) are team runs and do NOT
       count toward the batter's personal run tally.
    2. Balls Faced: Deliveries faced by the batter, EXCLUDING wide deliveries
       (`extras_type != 'wides'`). No-balls, byes, and leg-byes faced by the batter
       count as balls faced under standard ICC/T20 scoring rules.
    3. Dismissals: Count of deliveries where `player_dismissed == player` AND
       `dismissal_kind != 'retired hurt'`. Under MCC Law 25, retired hurt is
       'retired not out' and does not count as a lost wicket or batter dismissal.
       Run-outs of the non-striker are counted against the dismissed non-striker,
       not the facing striker. Tactical 'retired out' (MCC Law 25.4.3) is counted
       as a dismissal.
    4. Boundaries: Fours (`batsman_runs == 4`) and Sixes (`batsman_runs == 6`).
    5. Innings: Batting appearances, measured as unique (match_id, inning) occurrences
       where the player appeared at the crease as either striker or non-striker.
    6. Undefined rates:
       - Batting Average: `total_runs / dismissals` if `dismissals > 0`.
         If `dismissals == 0`, batting average is mathematically undefined and
         explicitly set to `np.nan` (or None). Downstream clustering safely filters
         or handles undefined averages without inventing synthetic statistics.
       - Strike Rate: `(total_runs / balls_faced) * 100.0` if `balls_faced > 0` else 0.0.
    """
    if deliveries_df.empty:
        return pd.DataFrame(
            columns=[
                "batter",
                "total_runs",
                "balls_faced",
                "innings",
                "dismissals",
                "fours",
                "sixes",
                "batting_average",
                "strike_rate",
                "boundary_runs",
                "boundary_pct",
            ]
        )

    df = deliveries_df.copy()

    # Ensure required columns are present
    required_cols = ["match_id", "inning", "batter", "batsman_runs"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Deliveries DataFrame missing required columns: {missing}")

    # Strip potential string whitespace
    for col in ["batter", "non_striker", "player_dismissed", "dismissal_kind", "extras_type"]:
        if col in df.columns and pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().replace({"None": None, "nan": None, "": None})

    # 1. Build comprehensive player index from union of batter, non_striker, and player_dismissed
    player_candidates = set(df["batter"].dropna())
    if "non_striker" in df.columns:
        player_candidates.update(df["non_striker"].dropna())
    if "player_dismissed" in df.columns:
        player_candidates.update(df["player_dismissed"].dropna())

    valid_players = sorted([
        p for p in player_candidates
        if p is not None and str(p).strip() != "" and str(p).strip().lower() not in ("none", "nan")
    ])
    player_index = pd.Index(valid_players, name="batter")

    # 2. Total runs off the bat per batter
    runs_series = df.groupby("batter")["batsman_runs"].sum().rename("total_runs")

    # 3. Balls faced: all deliveries to batter except wides
    if "extras_type" in df.columns:
        faced_mask = df["extras_type"] != "wides"
    else:
        faced_mask = pd.Series(True, index=df.index)
    balls_series = df[faced_mask].groupby("batter").size().rename("balls_faced")

    # 4. Innings count: unique (match_id, inning) appearances as striker or non-striker
    appearance_dfs = [df[["batter", "match_id", "inning"]].rename(columns={"batter": "player"})]
    if "non_striker" in df.columns:
        appearance_dfs.append(df[["non_striker", "match_id", "inning"]].rename(columns={"non_striker": "player"}))
    if "player_dismissed" in df.columns:
        appearance_dfs.append(df[["player_dismissed", "match_id", "inning"]].rename(columns={"player_dismissed": "player"}))

    appearances = pd.concat(appearance_dfs, ignore_index=True).dropna(subset=["player"])
    appearances = appearances[appearances["player"].isin(player_index)]

    innings_series = (
        appearances.drop_duplicates()
        .groupby("player")
        .size()
        .rename("innings")
    )

    # 5. Fours and Sixes off the bat
    fours_series = df[df["batsman_runs"] == 4].groupby("batter").size().rename("fours")
    sixes_series = df[df["batsman_runs"] == 6].groupby("batter").size().rename("sixes")

    # 6. Dismissals: player_dismissed and dismissal_kind != 'retired hurt'
    # Tactical 'retired out' is counted as a dismissal under Law 25.4.3
    if "player_dismissed" in df.columns:
        out_mask = df["player_dismissed"].notna()
        if "dismissal_kind" in df.columns:
            dismissal_kind_clean = df["dismissal_kind"].astype(str).str.strip().str.lower()
            out_mask = out_mask & (dismissal_kind_clean != "retired hurt")
        dismissals_series = (
            df[out_mask].groupby("player_dismissed").size().rename("dismissals")
        )
    else:
        dismissals_series = pd.Series(0, index=player_index, name="dismissals")

    # Combine into batter statistics DataFrame indexed by player_index
    stats = pd.DataFrame(index=player_index).join(
        [runs_series, balls_series, innings_series, dismissals_series, fours_series, sixes_series]
    ).fillna(0)

    # Convert integer columns
    int_cols = ["total_runs", "balls_faced", "innings", "dismissals", "fours", "sixes"]
    for c in int_cols:
        stats[c] = stats[c].astype(int)

    # Calculate derived rate metrics with explicit handling of undefined cases
    # Batting average: total_runs / dismissals; if dismissals == 0, undefined (np.nan)
    stats["batting_average"] = np.where(
        stats["dismissals"] > 0,
        (stats["total_runs"] / stats["dismissals"]).round(2),
        np.nan,
    )

    # Strike rate: (total_runs / balls_faced) * 100.0; if balls_faced == 0, safe default 0.0
    stats["strike_rate"] = np.where(
        stats["balls_faced"] > 0,
        ((stats["total_runs"] / stats["balls_faced"]) * 100.0).round(2),
        0.0,
    )

    # Derived boundary statistics
    stats["boundary_runs"] = stats["fours"] * 4 + stats["sixes"] * 6
    stats["boundary_pct"] = np.where(
        stats["total_runs"] > 0,
        ((stats["boundary_runs"] / stats["total_runs"]) * 100.0).round(2),
        0.0,
    )

    stats = stats.reset_index().rename(columns={"index": "batter"})
    return stats


def filter_batters(
    stats_df: pd.DataFrame,
    min_balls: int = DEFAULT_MIN_BALLS,
    min_innings: Optional[int] = None,
    drop_undefined_averages: bool = True,
) -> pd.DataFrame:
    """
    Filter batter statistics to include only players meeting minimum activity criteria.

    Rationale:
    In cricket analytics, players with very small sample sizes (e.g., tailenders facing
    5 balls, or batters dismissed for 0 after 1 delivery) exhibit extreme, volatile rates
    such as strike rates of 300+ or batting averages of 0/infinity. These outliers distort
    feature scaling and cluster centroids. Applying a minimum threshold (e.g., >= 200 balls)
    ensures that clustering operates on established batters with statistically reliable profiles.
    Batters with 0 dismissals (undefined batting average / NaN) are safely excluded by default
    to maintain mathematical validity without inventing synthetic statistics.
    """
    if stats_df.empty:
        return stats_df.copy()

    filtered = stats_df.copy()
    if min_balls > 0:
        filtered = filtered[filtered["balls_faced"] >= min_balls]
    if min_innings is not None and min_innings > 0:
        filtered = filtered[filtered["innings"] >= min_innings]
    if drop_undefined_averages and "batting_average" in filtered.columns:
        filtered = filtered[filtered["batting_average"].notna()]

    return filtered.reset_index(drop=True)


def evaluate_kmeans_clusters(
    stats_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    k_range: Tuple[int, int] = DEFAULT_K_RANGE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> Dict[str, Any]:
    """
    Evaluate K-Means clustering across a range of k values using the silhouette score.

    Parameters:
    - stats_df: DataFrame of batter statistics.
    - features: List of numeric feature columns to cluster on (default: batting_average, strike_rate).
    - k_range: Tuple of (min_k, max_k) cluster counts to evaluate.
    - random_state: Seed for reproducible centroid initialization.
    - n_init: Number of random initializations.

    Returns:
    Dictionary containing:
    - 'silhouette_scores': Dict mapping k to silhouette score.
    - 'best_k': The k with the highest silhouette score (or None if no valid candidate).
    - 'best_score': The highest silhouette score (or None).
    - 'scaler': Fitted StandardScaler instance.
    - 'X_scaled': Standardized feature matrix.
    - 'features': List of features used.
    """
    if features is None:
        features = list(DEFAULT_FEATURES)

    if stats_df.empty:
        return {
            "silhouette_scores": {},
            "best_k": None,
            "best_score": None,
            "scaler": None,
            "X_scaled": np.empty((0, len(features))),
            "models": {},
            "features": features,
        }

    missing_features = [f for f in features if f not in stats_df.columns]
    if missing_features:
        raise ValueError(f"Features missing from stats DataFrame: {missing_features}")

    # Safely filter out records with undefined (NaN) values in clustering features
    valid_mask = stats_df[features].notna().all(axis=1)
    clean_df = stats_df[valid_mask].copy() if not valid_mask.all() else stats_df

    if clean_df.empty:
        return {
            "silhouette_scores": {},
            "best_k": None,
            "best_score": None,
            "scaler": None,
            "X_scaled": np.empty((0, len(features))),
            "models": {},
            "features": features,
        }

    X = clean_df[features].values.astype(float)
    n_samples = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores: Dict[int, float] = {}
    models: Dict[int, KMeans] = {}

    min_k, max_k = k_range
    for k in range(min_k, max_k + 1):
        # Silhouette score requires at least k+1 samples and at least 2 distinct clusters
        if n_samples <= k:
            continue

        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)

        # Ensure more than 1 unique cluster was formed
        if len(set(labels)) > 1:
            score = float(silhouette_score(X_scaled, labels))
            silhouette_scores[k] = round(score, 4)
            models[k] = km

    if silhouette_scores:
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]
    else:
        best_k = None
        best_score = None

    return {
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_score": best_score,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "models": models,
        "features": features,
    }


def cluster_batters(
    stats_df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    features: Optional[List[str]] = None,
    k_range: Tuple[int, int] = DEFAULT_K_RANGE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[int, float]]:
    """
    Fit K-Means clustering on standardized batter features and assign cluster labels.

    If n_clusters is None, the candidate k with the best silhouette score within
    k_range is selected automatically.

    Returns:
    - clustered_df: Copy of stats_df with an added 'cluster' column.
    - model: Fitted KMeans instance for the chosen k.
    - scaler: Fitted StandardScaler instance.
    - silhouette_scores: Dictionary of silhouette scores evaluated for candidate k values.
    """
    if features is None:
        features = list(DEFAULT_FEATURES)

    # Safely filter out records with undefined (NaN) features before clustering
    valid_mask = stats_df[features].notna().all(axis=1)
    clean_df = stats_df[valid_mask].copy() if not valid_mask.all() else stats_df

    eval_result = evaluate_kmeans_clusters(
        stats_df=clean_df,
        features=features,
        k_range=k_range,
        random_state=random_state,
        n_init=n_init,
    )

    if n_clusters is None:
        chosen_k = eval_result["best_k"]
        if chosen_k is None:
            raise ValueError(
                f"Cannot determine optimal clusters: insufficient sample size ({len(clean_df)}) for k_range {k_range}."
            )
    else:
        chosen_k = n_clusters

    if len(clean_df) < chosen_k:
        raise ValueError(
            f"Sample size ({len(clean_df)}) is smaller than requested clusters ({chosen_k})."
        )

    scaler = eval_result["scaler"]
    X_scaled = eval_result["X_scaled"]

    if chosen_k in eval_result.get("models", {}):
        model = eval_result["models"][chosen_k]
        labels = model.labels_
    else:
        model = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)

    clustered_df = clean_df.copy().reset_index(drop=True)
    clustered_df["cluster"] = labels

    return clustered_df, model, scaler, eval_result["silhouette_scores"]


def summarize_clusters(
    clustered_df: pd.DataFrame,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Generate an interpretable statistical summary for each cluster.

    Computes player counts, mean and median batting average, strike rate,
    total runs, innings, and sample representative players per cluster.
    """
    if clustered_df.empty or cluster_col not in clustered_df.columns:
        return pd.DataFrame()

    summary_rows = []
    for cluster_id, group in clustered_df.groupby(cluster_col):
        # Identify top 3 run-scorers in this cluster as illustrative representatives
        top_scorers = group.sort_values(by="total_runs", ascending=False)["batter"].head(3).tolist()

        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "batter_count": int(len(group)),
                "mean_batting_average": round(float(group["batting_average"].mean()), 2),
                "median_batting_average": round(float(group["batting_average"].median()), 2),
                "mean_strike_rate": round(float(group["strike_rate"].mean()), 2),
                "median_strike_rate": round(float(group["strike_rate"].median()), 2),
                "mean_total_runs": round(float(group["total_runs"].mean()), 1),
                "mean_innings": round(float(group["innings"].mean()), 1),
                "mean_boundary_pct": round(float(group["boundary_pct"].mean()), 2)
                if "boundary_pct" in group.columns
                else 0.0,
                "representative_batters": ", ".join(top_scorers),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="cluster").reset_index(drop=True)
    return summary_df


def run_batter_clustering(
    matches_path: str = "Datasets/matches_2008-2024.csv",
    deliveries_path: str = "Datasets/deliveries_2008-2024.csv",
    max_season: Optional[int] = DEFAULT_SPLIT_SEASON,
    min_balls: int = DEFAULT_MIN_BALLS,
    n_clusters: Optional[int] = None,
    features: Optional[List[str]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end batter performance clustering workflow.

    Workflow:
    1. Loads and cleans matches and deliveries using project utilities without file mutation.
    2. Applies chronological cutoff if max_season is specified (default: 2021, matching
       the training cutoff used across regression, classification, and chase models).
    3. Aggregates delivery records into batter statistics with strict cricket accounting rules.
    4. Filters out small-sample batters below min_balls threshold.
    5. Standardizes features with StandardScaler and evaluates KMeans across k=2..5.
    6. Selects optimal cluster count based on silhouette score (or uses n_clusters if supplied).
    7. Computes and returns cluster assignments and descriptive summary statistics.
    """
    cleaned_matches, cleaned_deliveries, _ = load_and_preprocess_data(
        matches_path=matches_path,
        deliveries_path=deliveries_path,
    )

    # Apply chronological season cutoff if requested
    if max_season is not None and "season" in cleaned_matches.columns:
        valid_match_ids = set(cleaned_matches[cleaned_matches["season"] <= max_season]["id"])
        deliveries_subset = cleaned_deliveries[cleaned_deliveries["match_id"].isin(valid_match_ids)]
    else:
        deliveries_subset = cleaned_deliveries

    # 1. Aggregate statistics
    raw_stats = calculate_batter_statistics(deliveries_subset)

    # 2. Filter minimum sample threshold
    filtered_stats = filter_batters(raw_stats, min_balls=min_balls)

    if filtered_stats.empty:
        raise ValueError(f"No batters met the minimum threshold of {min_balls} balls faced.")

    # 3. Cluster and summarize
    clustered_df, model, scaler, silhouette_scores = cluster_batters(
        stats_df=filtered_stats,
        n_clusters=n_clusters,
        features=features,
        random_state=random_state,
    )

    summary_df = summarize_clusters(clustered_df)

    return {
        "clustered_df": clustered_df,
        "summary_df": summary_df,
        "silhouette_scores": silhouette_scores,
        "chosen_k": model.n_clusters,
        "model": model,
        "scaler": scaler,
        "features": features if features is not None else list(DEFAULT_FEATURES),
        "total_batters_evaluated": len(clustered_df),
        "max_season": max_season,
        "min_balls": min_balls,
    }
