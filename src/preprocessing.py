"""
Data loading and preprocessing utilities for the IPL Match Intelligence Lab.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd


class DataValidationError(ValueError):
    """Raised when dataset validation checks fail."""
    pass


TEAM_NAME_MAPPING: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

VENUE_NAME_MAPPING: Dict[str, str] = {
    # City suffix removals & formatting variants
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Sheikh Zayed Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Vidarbha Cricket Association Stadium, Jamtha": "Vidarbha Cricket Association Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    # Approved historical renamings
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    # Note: "Sardar Patel Stadium, Motera" is intentionally kept distinct from "Narendra Modi Stadium"
}


def normalize_venue(venue_name: Optional[str]) -> Optional[str]:
    """
    Normalize a venue name to its canonical representation.

    Rules:
    - Strips leading and trailing whitespace.
    - Applies canonical mapping for known city-suffixed, punctuated, or renamed variants.
    - Preserves missing values (None, NaN) safely.
    - Leaves unmapped or unknown venue names unchanged without guessing.
    """
    if venue_name is None or pd.isna(venue_name):
        return venue_name
    if not isinstance(venue_name, str):
        return venue_name

    cleaned = venue_name.strip()
    return VENUE_NAME_MAPPING.get(cleaned, cleaned)

REQUIRED_MATCHES_COLUMNS: List[str] = [
    "id",
    "season",
    "city",
    "date",
    "match_type",
    "player_of_match",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "result",
    "result_margin",
    "target_runs",
    "target_overs",
    "super_over",
    "method",
    "umpire1",
    "umpire2",
]

REQUIRED_DELIVERIES_COLUMNS: List[str] = [
    "match_id",
    "inning",
    "batting_team",
    "bowling_team",
    "over",
    "ball",
    "batter",
    "bowler",
    "non_striker",
    "batsman_runs",
    "extra_runs",
    "total_runs",
    "extras_type",
    "is_wicket",
    "player_dismissed",
    "dismissal_kind",
    "fielder",
]

PLACEHOLDER_STRINGS: Set[str] = {"NA", ""}

DELIVERIES_NULL_COLUMNS: List[str] = [
    "extras_type",
    "player_dismissed",
    "dismissal_kind",
    "fielder",
]


def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from column headers and string cells."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return cleaned


def _strip_and_clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from column headers and string cells,
    and convert verified placeholder strings ("NA", empty strings) to None.
    Preserves handling for datasets where all text columns may contain placeholders.
    """
    cleaned = _strip_whitespace(df)

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].replace({p: None for p in PLACEHOLDER_STRINGS})

    return cleaned


def clean_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean matches dataset:
    - Strip column and string whitespace
    - Replace placeholder strings with None/NaN
    - Standardize team names in team1, team2, toss_winner, winner
    - Standardize venue names to canonical representations
    - Preserve match ID data type (int64)
    """
    df = _strip_and_clean_strings(matches_df)

    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAPPING)

    if "venue" in df.columns:
        df["venue"] = df["venue"].apply(normalize_venue)

    if "id" in df.columns and not df["id"].isna().any():
        df["id"] = df["id"].astype("int64")

    return df


def clean_deliveries(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean deliveries dataset:
    - Strip column and string whitespace
    - Replace placeholder strings with None/NaN in designated delivery columns only
    - Standardize team names in batting_team, bowling_team
    - Preserve match ID data type (int64)
    """
    df = _strip_whitespace(deliveries_df)

    for col in DELIVERIES_NULL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace({p: None for p in PLACEHOLDER_STRINGS})

    team_cols = ["batting_team", "bowling_team"]
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAPPING)

    if "match_id" in df.columns and not df["match_id"].isna().any():
        df["match_id"] = df["match_id"].astype("int64")

    return df


def validate_datasets(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate cleaned datasets against integrity rules.
    Raises DataValidationError if critical issues are found.
    Returns a summary dictionary of validation metrics.
    """
    # 1. Check required columns in matches
    missing_matches_cols = set(REQUIRED_MATCHES_COLUMNS) - set(matches_df.columns)
    if missing_matches_cols:
        raise DataValidationError(f"Matches dataset missing required columns: {sorted(missing_matches_cols)}")

    # 2. Check required columns in deliveries
    missing_deliveries_cols = set(REQUIRED_DELIVERIES_COLUMNS) - set(deliveries_df.columns)
    if missing_deliveries_cols:
        raise DataValidationError(f"Deliveries dataset missing required columns: {sorted(missing_deliveries_cols)}")

    # 3. Check missing match IDs
    if matches_df["id"].isna().any():
        raise DataValidationError("Matches dataset contains missing (null) match IDs.")
    if deliveries_df["match_id"].isna().any():
        raise DataValidationError("Deliveries dataset contains missing (null) match IDs.")

    # 4. Check duplicate match IDs in matches
    duplicates_count = int(matches_df["id"].duplicated().sum())
    if duplicates_count > 0:
        raise DataValidationError(f"Matches dataset contains {duplicates_count} duplicate match IDs.")

    # 5. Check delivery match IDs that do not exist in matches
    matches_ids = set(matches_df["id"].unique())
    delivery_ids = set(deliveries_df["match_id"].unique())
    unmatched_ids = delivery_ids - matches_ids
    if unmatched_ids:
        raise DataValidationError(
            f"Found {len(unmatched_ids)} match IDs in deliveries that do not exist in matches: {sorted(unmatched_ids)[:5]}..."
        )

    return {
        "matches_count": len(matches_df),
        "deliveries_count": len(deliveries_df),
        "unique_matches_in_matches": len(matches_ids),
        "unique_matches_in_deliveries": len(delivery_ids),
        "unmatched_delivery_match_ids_count": len(unmatched_ids),
        "duplicate_match_ids_count": duplicates_count,
    }


def resolve_data_path(preferred_path: str, fallback_path: Optional[str] = None) -> Path:
    """Resolve a file path, checking preferred location then fallback location."""
    p = Path(preferred_path)
    if p.exists():
        return p
    if fallback_path:
        fb = Path(fallback_path)
        if fb.exists():
            return fb
    raise FileNotFoundError(
        f"Could not find dataset at '{preferred_path}'" + (f" or '{fallback_path}'" if fallback_path else "")
    )


def load_and_preprocess_data(
    matches_path: str = "data/matches_2008-2024.csv",
    deliveries_path: str = "data/deliveries_2008-2024.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load raw CSVs, apply cleaning and standardization, and validate.
    Returns: (cleaned_matches_df, cleaned_deliveries_df, audit_report_dict)
    """
    m_path = resolve_data_path(matches_path, matches_path.replace("data/", "Datasets/"))
    d_path = resolve_data_path(deliveries_path, deliveries_path.replace("data/", "Datasets/"))

    raw_matches = pd.read_csv(m_path, low_memory=False)
    raw_deliveries = pd.read_csv(d_path, low_memory=False)

    rows_matches_before = len(raw_matches)
    rows_deliveries_before = len(raw_deliveries)

    # Extract team names before standardization (strip whitespace first)
    raw_m_t1 = raw_matches.get("team1", pd.Series(dtype=object)).dropna().astype(str).str.strip()
    raw_m_t2 = raw_matches.get("team2", pd.Series(dtype=object)).dropna().astype(str).str.strip()
    teams_m_before = sorted(set(raw_m_t1) | set(raw_m_t2))

    # For raw deliveries, column names may have whitespace, so locate columns flexibly
    del_cols_stripped = [c.strip() for c in raw_deliveries.columns]
    bat_idx = del_cols_stripped.index("batting_team") if "batting_team" in del_cols_stripped else 2
    bowl_idx = del_cols_stripped.index("bowling_team") if "bowling_team" in del_cols_stripped else 3
    raw_d_bat = raw_deliveries.iloc[:, bat_idx].dropna().astype(str).str.strip()
    raw_d_bowl = raw_deliveries.iloc[:, bowl_idx].dropna().astype(str).str.strip()
    teams_d_before = sorted(set(raw_d_bat) | set(raw_d_bowl))

    cleaned_matches = clean_matches(raw_matches)
    cleaned_deliveries = clean_deliveries(raw_deliveries)

    teams_m_after = sorted(set(cleaned_matches["team1"].dropna()) | set(cleaned_matches["team2"].dropna()))
    teams_d_after = sorted(
        set(cleaned_deliveries["batting_team"].dropna()) | set(cleaned_deliveries["bowling_team"].dropna())
    )

    validation_info = validate_datasets(cleaned_matches, cleaned_deliveries)

    audit_report = {
        "rows_matches_before": rows_matches_before,
        "rows_matches_after": len(cleaned_matches),
        "rows_deliveries_before": rows_deliveries_before,
        "rows_deliveries_after": len(cleaned_deliveries),
        "unmatched_delivery_match_ids": validation_info["unmatched_delivery_match_ids_count"],
        "teams_matches_before": teams_m_before,
        "teams_matches_after": teams_m_after,
        "teams_deliveries_before": teams_d_before,
        "teams_deliveries_after": teams_d_after,
        "validation_info": validation_info,
    }

    return cleaned_matches, cleaned_deliveries, audit_report
