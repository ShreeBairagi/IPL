"""
Streamlit Dashboard for the IPL Match Intelligence Lab.

Interactive demonstration of:
1. Project Overview & Data Pipeline Metrics
2. First-Innings Final-Score Regression & Classification at 10 Overs
3. Second-Innings Chase-Win Probability Prediction at 10 Overs
4. Held-Out Model Evaluation & Verification
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.batter_clustering import run_batter_clustering
from src.dashboard_utils import (
    build_chase_input_df,
    build_first_innings_input_df,
    get_project_overview_data,
    get_teams_and_venues,
    load_all_models_and_metrics,
    validate_chase_inputs,
    validate_first_innings_inputs,
)

# Page configuration
st.set_page_config(
    page_title="IPL Match Intelligence Lab",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_cached_overview():
    """Cache dataset overview statistics to avoid reloading CSVs on every interaction."""
    return get_project_overview_data()


@st.cache_data
def get_cached_teams_and_venues():
    """Cache unique team and venue lists."""
    return get_teams_and_venues()


@st.cache_resource
def get_cached_models_and_metrics():
    """Cache trained model pipelines and held-out test evaluations."""
    return load_all_models_and_metrics()


@st.cache_resource
def get_cached_batter_clustering():
    """Cache batter clustering execution results using the existing module."""
    return run_batter_clustering()


def main():
    st.title("🏏 IPL Match Intelligence Lab")
    st.caption("10-Over Mid-Innings Forecasting & Chase-Win Probability Estimation")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Section:",
        [
            "1. Project Overview",
            "2. Final Score Prediction & Class",
            "3. Chase-Win Probability",
            "4. Model Evaluation",
            "5. Batter Performance Clustering",
        ],
    )

    teams, venues = get_cached_teams_and_venues()

    # SECTION 1: PROJECT OVERVIEW
    if section == "1. Project Overview":
        st.header("Project Overview & Methodology")
        overview = get_cached_overview()

        st.markdown(
            """
            The **IPL Match Intelligence Lab** analyzes regular 20-over Indian Premier League
            matches from **2008 to 2024**. Using match states extracted at exactly **10 completed overs**
            (overs 0–9), the lab evaluates three distinct forecasting tasks:
            1. **First-Innings Final Score Regression:** Predicting the final 20-over score.
            2. **Score Classification:** Categorizing final scores into *Low*, *Medium*, or *High*.
            3. **Chase-Win Probability:** Estimating the probability that the team batting second successfully chases the target.
            """
        )

        st.subheader("Strict Chronological Partitioning")
        st.info(
            "To prevent temporal data leakage and simulate genuine pre-match / in-play forecasting, "
            "seasons **2008–2021** serve as the training set, while modern seasons **2022–2024** "
            "serve as the held-out test set. All snapshots from a match remain strictly within a single partition."
        )

        st.subheader("Dataset & Snapshot Counts")
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Deliveries", f"{overview['raw_deliveries_count']:,}")
        col2.metric("Raw Matches", f"{overview['raw_matches_count']:,}")
        col3.metric("Cleaned 20-Over Matches", f"{overview['cleaned_matches_count']:,}")

        col4, col5 = st.columns(2)
        with col4:
            st.markdown("##### First-Innings 10-Over Snapshots")
            st.write(f"- **Total Eligible:** {overview['first_innings_total']}")
            st.write(f"- **Train (2008–2021):** {overview['first_innings_train']}")
            st.write(f"- **Test (2022–2024):** {overview['first_innings_test']}")

        with col5:
            st.markdown("##### Second-Innings (Chase) 10-Over Snapshots")
            st.write(f"- **Total Eligible:** {overview['second_innings_total']}")
            st.write(f"- **Train (2008–2021):** {overview['second_innings_train']}")
            st.write(f"- **Test (2022–2024):** {overview['second_innings_test']}")

    # SECTION 2: FINAL SCORE PREDICTION & CLASSIFICATION
    elif section == "2. Final Score Prediction & Class":
        st.header("First-Innings Final Score Prediction & Classification")
        st.write("Enter the match state after **exactly 10 completed overs** of the first innings:")

        with st.form("first_innings_form"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                batting_team = st.selectbox("Batting Team (1st Innings)", teams, index=0)
            with col_t2:
                # Default bowling team to second team in list
                default_bowl_idx = 1 if len(teams) > 1 else 0
                bowling_team = st.selectbox("Bowling Team", teams, index=default_bowl_idx)

            venue = st.selectbox("Venue / Stadium", venues, index=0)

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                current_score = st.number_input("Score at 10 Overs", min_value=0, max_value=300, value=85, step=1)
            with col_s2:
                wickets = st.number_input("Wickets Lost (0–10)", min_value=0, max_value=10, value=2, step=1)
            with col_s3:
                runs_last_5 = st.number_input("Runs in Overs 5–9", min_value=0, max_value=150, value=42, step=1)
            with col_s4:
                wickets_last_5 = st.number_input("Wickets in Overs 5–9", min_value=0, max_value=10, value=1, step=1)

            reg_model_choice = st.selectbox(
                "Regression Estimator",
                ["Ridge Regression", "Random Forest", "Linear Regression", "Gradient Boosting"],
                index=0,
            )

            submit_reg = st.form_submit_button("Predict Final Score & Class", use_container_width=True)

        if submit_reg:
            is_valid, err = validate_first_innings_inputs(
                current_score, wickets, runs_last_5, wickets_last_5, batting_team, bowling_team
            )
            if not is_valid:
                st.error(err)
            else:
                try:
                    data_bundle = get_cached_models_and_metrics()
                    reg_pipeline = data_bundle["regression"]["models"][reg_model_choice]["pipeline"]
                    thresholds = data_bundle["classification"]["thresholds"]
                    t_low = thresholds["threshold_low"]
                    t_high = thresholds["threshold_high"]

                    input_df = build_first_innings_input_df(
                        batting_team, bowling_team, venue, current_score, wickets, runs_last_5, wickets_last_5
                    )

                    predicted_score = float(reg_pipeline.predict(input_df)[0])
                    predicted_class = "Low" if predicted_score <= t_low else ("Medium" if predicted_score <= t_high else "High")

                    st.success("### Prediction Generated")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(
                            label=f"Predicted Final Score ({reg_model_choice})",
                            value=f"{predicted_score:.1f} runs",
                        )
                        st.caption("⚠️ **Model Estimate:** Mid-innings projection with ±17.8 MAE test uncertainty.")

                    with res_col2:
                        st.metric(
                            label="Predicted Score Class (Regression-then-cut)",
                            value=f"{predicted_class}",
                        )
                        st.caption(
                            f"Thresholds derived from training data: Low (≤{t_low:.0f}), "
                            f"Medium ({t_low:.0f}–{t_high:.0f}), High (>{t_high:.0f})."
                        )
                except Exception as e:
                    st.error(f"Error generating score prediction: {e}")

    # SECTION 3: CHASE-WIN PROBABILITY
    elif section == "3. Chase-Win Probability":
        st.header("Second-Innings Chase-Win Probability Prediction")
        st.write("Enter the match state after **exactly 10 completed overs** of the second innings chase:")

        with st.form("chase_form"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                chasing_team = st.selectbox("Chasing Team (Batting 2nd)", teams, index=0)
            with col_c2:
                default_def_idx = 1 if len(teams) > 1 else 0
                defending_team = st.selectbox("Defending Team (Bowling 2nd)", teams, index=default_def_idx)

            venue_chase = st.selectbox("Venue / Stadium", venues, index=0)

            col_ch1, col_ch2, col_ch3, col_ch4, col_ch5 = st.columns(5)
            with col_ch1:
                target = st.number_input("Target to Win", min_value=1, max_value=350, value=175, step=1)
            with col_ch2:
                current_score = st.number_input("Score at 10 Overs", min_value=0, max_value=300, value=85, step=1)
            with col_ch3:
                wickets = st.number_input("Wickets Lost (0–10)", min_value=0, max_value=10, value=2, step=1)
            with col_ch4:
                runs_last_5 = st.number_input("Runs in Overs 5–9", min_value=0, max_value=150, value=40, step=1)
            with col_ch5:
                wickets_last_5 = st.number_input("Wickets in Overs 5–9", min_value=0, max_value=10, value=1, step=1)

            chase_model_choice = st.selectbox(
                "Probabilistic Classifier",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", "K-Nearest Neighbors"],
                index=0,
            )

            submit_chase = st.form_submit_button("Estimate Chase-Win Probability", use_container_width=True)

        if submit_chase:
            is_valid, err = validate_chase_inputs(
                target, current_score, wickets, runs_last_5, wickets_last_5, chasing_team, defending_team
            )
            if not is_valid:
                st.error(err)
            else:
                try:
                    data_bundle = get_cached_models_and_metrics()
                    chase_pipeline = data_bundle["chase"]["models"][chase_model_choice]["pipeline"]

                    input_df = build_chase_input_df(
                        chasing_team, defending_team, venue_chase, target, current_score, wickets, runs_last_5, wickets_last_5
                    )

                    win_prob = float(chase_pipeline.predict_proba(input_df)[0, 1])
                    runs_needed = target - current_score
                    rrr = runs_needed / 10.0
                    crr = current_score / 10.0

                    st.success("### Chase Situation & Win Probability")
                    st.progress(win_prob)

                    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                    pcol1.metric("Win Probability (Chasing)", f"{win_prob * 100:.1f}%")
                    pcol2.metric("Runs Needed", f"{runs_needed} off 60 balls")
                    pcol3.metric("Required Run Rate", f"{rrr:.2f}")
                    pcol4.metric("Current Run Rate", f"{crr:.2f}")

                    st.caption(
                        "⚠️ **Model Estimate:** Probability output from "
                        f"**{chase_model_choice}** (test Brier score ~0.154). "
                        "Note: Reflects raw model confidence based on mid-match state; not formally calibrated."
                    )
                except Exception as e:
                    st.error(f"Error estimating chase probability: {e}")

    # SECTION 4: MODEL EVALUATION
    elif section == "4. Model Evaluation":
        st.header("Held-Out Test Set Model Evaluation")
        st.caption("All metrics measured strictly on held-out test seasons (2022–2024). No invented figures.")

        try:
            data_bundle = get_cached_models_and_metrics()
        except Exception as e:
            st.error(f"Failed to load evaluation metrics: {e}")
            return

        tab1, tab2, tab3 = st.tabs([
            "1. First-Innings Regression",
            "2. First-Innings Classification",
            "3. Chase-Win Probability",
        ])

        with tab1:
            st.subheader("First-Innings Regression (MAE & R²)")
            st.write("Chronological test set: seasons 2022–2024 ($N = 215$ matches).")
            reg_bundle = data_bundle.get("regression", {}) if isinstance(data_bundle, dict) else {}
            reg_metrics = reg_bundle.get("metrics")
            if not reg_metrics:
                st.warning("No regression evaluation metrics available.")
            else:
                reg_rows = [
                    {"Model": m, "Test MAE (Runs)": d.get("mae", float("nan")), "Test R²": d.get("r2", float("nan"))}
                    for m, d in reg_metrics.items()
                ]
                st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)

        with tab2:
            st.subheader("First-Innings Score Classification")
            st.write(
                f"Training-derived thresholds: Low (≤152), Medium (153–175), High (>175). "
                f"Test set: seasons 2022–2024 ($N = 215$ matches)."
            )
            clf_bundle = data_bundle.get("classification", {}) if isinstance(data_bundle, dict) else {}
            clf_models = clf_bundle.get("models")
            if not clf_models:
                st.warning("No classification evaluation metrics available.")
            else:
                clf_rows = [
                    {
                        "Model": m,
                        "Accuracy": d.get("accuracy", float("nan")),
                        "Macro F1": d.get("macro_f1", float("nan")),
                        "Confusion Matrix [Low, Med, High]": str(d.get("confusion_matrix", "N/A")),
                    }
                    for m, d in clf_models.items()
                ]
                st.dataframe(pd.DataFrame(clf_rows), use_container_width=True)

        with tab3:
            st.subheader("Chase-Win Probability Prediction")
            st.write("Chronological test set: seasons 2022–2024 ($N = 212$ matches: 110 defending wins, 102 chasing wins).")
            chase_bundle = data_bundle.get("chase", {}) if isinstance(data_bundle, dict) else {}
            chase_models = chase_bundle.get("models")
            if not chase_models:
                st.warning("No chase-win probability evaluation metrics available.")
            else:
                chase_rows = [
                    {
                        "Model": m,
                        "Accuracy": d.get("accuracy", float("nan")),
                        "Precision": d.get("precision", float("nan")),
                        "Recall": d.get("recall", float("nan")),
                        "F1-Score": d.get("f1", float("nan")),
                        "ROC-AUC": d.get("roc_auc", float("nan")),
                        "Brier Score": d.get("brier_score", float("nan")),
                        "Confusion Matrix [[TN, FP], [FN, TP]]": str(d.get("confusion_matrix", "N/A")),
                    }
                    for m, d in chase_models.items()
                ]
                st.dataframe(pd.DataFrame(chase_rows), use_container_width=True)

    # SECTION 5: BATTER PERFORMANCE CLUSTERING
    elif section == "5. Batter Performance Clustering":
        st.header("Batter Performance Clustering")
        st.caption(
            "Unsupervised K-Means clustering identifying empirical T20 batting archetypes "
            "based on historical delivery data (2008–2021, min. 200 balls faced)."
        )

        try:
            clustering_result = get_cached_batter_clustering()
            clustered_df = clustering_result.get("clustered_df")
            summary_df = clustering_result.get("summary_df")
            scores = clustering_result.get("silhouette_scores", {})
            chosen_k = clustering_result.get("chosen_k")
            total_batters = clustering_result.get("total_batters_evaluated", 0)

            if clustered_df is None or clustered_df.empty or chosen_k is None:
                st.warning("No valid clusters could be formed from the dataset.")
                return

            # 1. Summary Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Eligible Batters Evaluated", f"{total_batters}")
            col_m2.metric("Optimal Clusters (k)", f"{chosen_k}")
            best_score = scores.get(chosen_k, float("nan"))
            col_m3.metric("Selected Silhouette Score", f"{best_score:.4f}" if not pd.isna(best_score) else "N/A")

            # 2. Candidate K Silhouette Evaluation Table
            st.subheader("Candidate Cluster Evaluation (k = 2 to 5)")
            st.write(
                "K-Means was evaluated across candidate cluster counts $k \\in [2, 5]$ "
                "on standardized features (`batting_average`, `strike_rate`). "
                "The configuration maximizing the silhouette score was selected as optimal:"
            )
            scores_rows = [
                {
                    "Candidate Clusters (k)": k,
                    "Silhouette Score": f"{score:.4f}",
                    "Status": "✅ Selected Optimal" if k == chosen_k else "Evaluated",
                }
                for k, score in sorted(scores.items())
            ]
            st.dataframe(pd.DataFrame(scores_rows), use_container_width=True, hide_index=True)

            # 3. Cluster Summary Table
            st.subheader("Cluster Profiles & Aggregate Statistics")
            st.write(
                "Aggregated performance statistics computed directly from historical records "
                "for each empirical cluster:"
            )
            summary_display = summary_df.copy()
            rename_map = {
                "cluster": "Cluster ID",
                "batter_count": "Batter Count",
                "mean_batting_average": "Mean Batting Avg",
                "median_batting_average": "Median Batting Avg",
                "mean_strike_rate": "Mean Strike Rate",
                "median_strike_rate": "Median Strike Rate",
                "mean_total_runs": "Mean Total Runs",
                "mean_innings": "Mean Innings",
                "mean_boundary_pct": "Mean Boundary %",
                "representative_batters": "Top Run-Scorers (Illustrative)",
            }
            summary_display = summary_display.rename(columns={k: v for k, v in rename_map.items() if k in summary_display.columns})
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

            # 4. Scatter Plot: Batting Average vs. Strike Rate
            st.subheader("Clustering Feature Space")
            st.write(
                "Scatter visualization of all eligible batters across the primary clustering dimensions. "
                "Points are colored by their assigned statistical cluster:"
            )

            fig, ax = plt.subplots(figsize=(9, 5.5))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for cluster_id, group in clustered_df.groupby("cluster"):
                color = colors[int(cluster_id) % len(colors)]
                ax.scatter(
                    group["batting_average"],
                    group["strike_rate"],
                    label=f"Cluster {cluster_id} (n={len(group)})",
                    color=color,
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=0.5,
                    s=55,
                )

            ax.set_title("IPL Batter Clustering: Batting Average vs. Strike Rate", fontsize=12, fontweight="bold")
            ax.set_xlabel("Batting Average (Runs per Dismissal)", fontsize=10)
            ax.set_ylabel("Strike Rate (Runs per 100 Balls)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(title="Cluster Assignment", loc="best", framealpha=0.9)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # 5. Full Batter Assignments Table
            st.subheader("Assigned Player Profiles")
            st.write("Browse all qualified batters, their assigned cluster, and underlying career statistics:")

            cluster_options = ["All Clusters"] + [f"Cluster {c}" for c in sorted(clustered_df["cluster"].unique())]
            chosen_filter = st.selectbox("Filter by Cluster:", cluster_options, index=0)

            filtered_batters = clustered_df.copy()
            if chosen_filter != "All Clusters":
                target_c = int(chosen_filter.replace("Cluster ", ""))
                filtered_batters = filtered_batters[filtered_batters["cluster"] == target_c]

            display_cols = [
                "batter",
                "cluster",
                "batting_average",
                "strike_rate",
                "total_runs",
                "balls_faced",
                "innings",
                "dismissals",
                "fours",
                "sixes",
                "boundary_pct",
            ]
            cols_to_show = [c for c in display_cols if c in filtered_batters.columns]
            table_to_show = filtered_batters[cols_to_show].sort_values(
                by=["cluster", "total_runs"], ascending=[True, False]
            )
            st.dataframe(table_to_show, use_container_width=True, hide_index=True)

            # 6. Explanatory Context & Methodological Distinction
            st.markdown("---")
            st.subheader("Methodology & Cricket Analytics Interpretation")
            st.markdown(
                """
                - **Unsupervised Learning (K-Means):** Unlike first-innings regression or chase classification—which train against historical match outcomes—K-Means is an unsupervised clustering algorithm. It discovers natural geometric groupings among players in feature space without human-assigned target labels.
                - **Feature Standardization (StandardScaler):** Batting average (typically ~15–40 runs per dismissal) and strike rate (typically ~100–160 runs per 100 balls) reside on vastly different numeric scales. If unscaled, Euclidean distance calculations in K-Means would be dominated almost entirely by strike rate variance. Standardization transforms both features to zero mean and unit variance ($z = (x - \\mu)/\\sigma$), giving both dimensions equal weight during distance minimization.
                - **Silhouette Score Interpretation:** The silhouette score measures both how tightly grouped points are within their assigned cluster (cohesion) and how well-separated they are from neighboring clusters (separation), ranging from $-1.0$ (poor/misclustered) to $+1.0$ (perfect isolation). The candidate $k$ maximizing this score within $k \\in [2, 5]$ was programmatically selected.
                - **Statistical Archetypes vs. Official Rankings:** Cluster IDs (`0`, `1`, etc.) are arbitrary categorical groupings reflecting empirical stylistic archetypes (e.g., high-volume anchors vs. lower-order anchors/accumulators). They do **not** represent official ICC/IPL rankings, subjective player quality, or value judgments.
                - **Eligibility Threshold & Chronological Cutoff:**
                  - **Minimum Balls Faced ($\\ge 200$ balls):** Excludes tailenders and short-sample appearances whose small-sample rate volatility would distort cluster centroids.
                  - **Season Cutoff (seasons $\\le 2021$):** Restricts historical data to matches through the 2021 season, maintaining strict chronological consistency with the training window of the regression, classification, and chase prediction models without future data leakage.
                """
            )

        except Exception as e:
            st.error(f"Error loading or computing batter clustering: {e}")


if __name__ == "__main__":
    main()
