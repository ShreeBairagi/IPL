
"""
Root alias forwarding to src.batter_clustering for seamless import compatibility.
"""
import sys
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.batter_clustering import (
    DEFAULT_FEATURES,
    DEFAULT_K_RANGE,
    DEFAULT_MIN_BALLS,
    DEFAULT_N_INIT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_SEASON,
    calculate_batter_statistics,
    cluster_batters,
    evaluate_kmeans_clusters,
    filter_batters,
    run_batter_clustering,
    summarize_clusters,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_K_RANGE",
    "DEFAULT_MIN_BALLS",
    "DEFAULT_N_INIT",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_SPLIT_SEASON",
    "calculate_batter_statistics",
    "cluster_batters",
    "evaluate_kmeans_clusters",
    "filter_batters",
    "run_batter_clustering",
    "summarize_clusters",
]
