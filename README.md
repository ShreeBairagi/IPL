# IPL Match Intelligence Lab

An end-to-end, leak-free machine learning and predictive analytics system for the Indian Premier League (IPL), built on ball-by-ball delivery data from the 2008 through 2024 seasons.

The system evaluates mid-innings match states at exactly 10 completed overs (overs 0–9) to forecast first-innings final scores, classify score tiers, estimate second-innings chase-win probabilities, compute dynamic stadium-specific par scores, and cluster historical batter profiles using unsupervised learning.

---

## 1. Project Architecture & Predictive Tasks

```
IPL Match Intelligence Lab
├── 1. Data Ingestion & Preprocessing (data/ & src/preprocessing.py)
│     └── Standardizes team franchises & venues; validates relational integrity across 1,095 matches & 260,920 deliveries.
├── 2. 10-Over Snapshot Extraction (src/snapshots.py)
│     ├── First-Innings Snapshots: Overs 0–9 match state, last 5 overs momentum, final score target.
│     └── Second-Innings Snapshots: Target, current score, wickets, required & current run rates, target achieved.
├── 3. Supervised Machine Learning Pipelines
│     ├── Final Score Regression (src/regression.py): Predicts continuous 20-over first-innings final score.
│     ├── Score Discretization & Classification (src/classification.py): Categorizes final scores into Low, Medium, High.
│     └── Chase-Win Probability Estimation (src/chase.py): Predicts probability that chasing team wins.
├── 4. Dynamic Stadium-Specific Par-Score Estimation (src/par_score.py & data/stadium_par_scores.csv)
│     └── Evaluates continuous regression predictions against empirical ground tertiles with Bayesian shrinkage.
├── 5. Unsupervised Batter Clustering (src/batter_clustering.py)
│     └── K-Means clustering (k=2 to 5, optimal k=2) on historical players (seasons <= 2021) based on scoring efficiency.
├── 6. Feature & Model Ablation Audits
│     ├── Toss Feature Impact Evaluation: Baseline vs. toss-enhanced comparison across all tasks.
│     └── XGBoost Integration (XGBRegressor & MulticlassXGBClassifier): Standardized benchmark against existing models.
└── 7. Interactive Streamlit Dashboard (app.py & src/dashboard_utils.py)
      └── 5-section interactive analytics interface with input validation, live inference, and clustering visuals.
```

---

## 2. Chronological Partitioning & Leakage Prevention

To ensure strict real-world generalization and prevent temporal data leakage:
- **Training Partition**: Matches from IPL seasons **2008 through 2021** (seasons $\le 2021$).
- **Held-Out Test Partition**: Matches from IPL seasons **2022 through 2024** (seasons $\ge 2022$).
- **Partition Integrity Safeguards**:
  - **Zero match overlap** between training and test sets.
  - Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) are fitted strictly on training data inside scikit-learn `Pipeline` objects.
  - Unseen franchise teams (e.g., Gujarat Titans, Lucknow Super Giants) and venues in 2022+ are handled safely via `handle_unknown="ignore"`.
  - Target variables (`final_score`, `target_achieved`), post-match metadata (`winner`, `result`), and tracking IDs (`match_id`, `season`) are strictly excluded from feature matrices $X$.
  - Global score classification thresholds ($152.0$ and $175.0$ runs) are calculated exclusively from training set tertiles ($33.33\%$ and $66.67\%$).
  - Stadium par scores and historical distributions are derived strictly from matches with `season <= 2021`; unseen grounds in test seasons fall back to competition training baselines without inspecting future matches.
  - Batter clustering eligibility and rate statistics are computed strictly on deliveries through the 2021 season.

---

## 3. Feature Configurations & Evaluated Models

### Feature Schemas (at 10 completed overs)

- **First-Innings Features**:
  - *Numeric (4)*: `current_score`, `wickets`, `runs_last_5`, `wickets_last_5`
  - *Categorical (3)*: `batting_team`, `bowling_team`, `venue`
  - *Optional Toss (2)*: `toss_winner`, `toss_decision` (when `include_toss=True`)
- **Second-Innings Chase Features**:
  - *Numeric (8)*: `target`, `current_score`, `wickets`, `runs_last_5`, `wickets_last_5`, `runs_needed`, `required_run_rate`, `current_run_rate`
  - *Categorical (3)*: `batting_team`, `bowling_team`, `venue`
  - *Optional Toss (2)*: `toss_winner`, `toss_decision` (when `include_toss=True`)

### Model Summary Across Tasks

| Task | Estimators Evaluated | Evaluation Metrics |
| :--- | :--- | :--- |
| **First-Innings Regression** | Mean Baseline, Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost (`XGBRegressor`) | MAE, RMSE, $R^2$ |
| **First-Innings Classification** | Majority Baseline, K-Nearest Neighbors, Decision Tree, Support Vector Machine, Random Forest, XGBoost (`MulticlassXGBClassifier`), Regression-then-cut (Ridge, RF) | Accuracy, Macro F1, Confusion Matrix |
| **Chase-Win Probability** | Majority Baseline, Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, XGBoost (`XGBClassifier`) | Accuracy, ROC-AUC, F1-Score, Brier Score |
| **Stadium Par Classification** | Transparent Statistical Discretization (Par Differential Bands $\pm 6.0$ Runs) | Par Difference ($\Delta$), Ground Relative Tier (Below Par, Par, Above Par) |
| **Batter Clustering** | K-Means ($k=2$ to $5$, evaluated with `n_init=10`, `random_state=42`) | Silhouette Score, Cluster Centroids |

---

## 4. Benchmark Evaluation Results (Test Seasons 2022–2024)

### Task 1: First-Innings Final Score Regression (Test $N=215$)
| Model | Baseline MAE | Baseline $R^2$ | With Toss MAE | With Toss $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| Mean Baseline | 30.14 | -0.3052 | 30.14 | -0.3052 |
| Linear Regression | **17.68** | 0.5473 | **17.67** | 0.5507 |
| Ridge Regression | 17.69 | 0.5473 | 17.68 | 0.5495 |
| Random Forest | 18.08 | 0.5463 | 17.97 | 0.5515 |
| Gradient Boosting | 18.06 | **0.5575** | 18.04 | **0.5625** |
| XGBoost | 19.51 | 0.4666 | 19.45 | 0.4522 |

### Task 2: First-Innings Score Classification (Test $N=215$)
| Model | Baseline Accuracy | Baseline Macro F1 | With Toss Accuracy | With Toss Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| Majority Baseline | 0.1814 | 0.1024 | 0.1814 | 0.1024 |
| K-Nearest Neighbors | 0.6326 | 0.5738 | 0.6047 | 0.5526 |
| Decision Tree | 0.5674 | 0.5165 | 0.5907 | 0.5570 |
| Support Vector Machine | 0.6837 | **0.6375** | 0.6465 | 0.5897 |
| Random Forest | **0.6884** | 0.6228 | **0.6605** | **0.6045** |
| XGBoost | 0.6512 | 0.5975 | 0.6512 | 0.6002 |
| Regression-then-cut (Ridge) | 0.6651 | 0.6133 | 0.6651 | 0.6133 |
| Regression-then-cut (RF) | 0.6512 | 0.5959 | 0.6698 | 0.6186 |

### Task 3: Second-Innings Chase-Win Probability (Test $N=212$)
| Model | Baseline Accuracy | Baseline ROC-AUC | Baseline Brier | With Toss Accuracy | With Toss ROC-AUC | With Toss Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | 0.4811 | 0.5000 | 0.5189 | 0.4811 | 0.5000 | 0.5189 |
| Logistic Regression | 0.7877 | **0.8524** | 0.1580 | 0.8019 | 0.8545 | 0.1571 |
| K-Nearest Neighbors | 0.7217 | 0.7954 | 0.1919 | 0.7170 | 0.7749 | 0.2025 |
| Random Forest | 0.7783 | 0.8411 | 0.1608 | 0.7830 | 0.8439 | 0.1594 |
| Gradient Boosting | **0.7972** | 0.8513 | **0.1561** | **0.8113** | **0.8557** | **0.1512** |
| XGBoost | 0.7500 | 0.8275 | 0.1901 | 0.7547 | 0.8325 | 0.1824 |

---

## 5. Dynamic Stadium-Specific Par-Score Estimation

In cricket analytics, universal score thresholds fail to reflect ground characteristics: a total of $175$ at the high-scoring **M Chinnaswamy Stadium (Bengaluru)** is merely par, whereas $175$ at **MA Chidambaram Stadium (Chepauk, Chennai)** is a match-winning high score.

The system features dynamic stadium-specific par evaluation (`src/par_score.py`) without introducing an extra ML classification model.

### Methodology & Workflow

1. **Continuous Score Prediction**: The user's input state is evaluated by the existing 10-over Regression pipeline (e.g., Ridge or Random Forest) to produce a projected first-innings score $\hat{y}$.
2. **Empirical Distribution Aggregation**: Eligible 20-over first innings from historical matches (`season <= 2021`) are aggregated per venue to compute sample size ($N$), median score ($\tilde{x}_{\text{venue}}$), mean score ($\bar{x}_{\text{venue}}$), standard deviation, recent 4-season scoring trend ($\bar{x}_{\text{recent}}$), and tertile percentiles ($P_{33}, P_{66}$).
3. **Empirical Bayes Shrinkage for Small-Sample Grounds**:
   To prevent volatility for grounds with few matches (e.g., *OUTsurance Oval* with $N=2$), baseline statistics are shrunk toward the competition training baseline ($\tilde{x}_{\text{competition}} = 163.5$) using credibility weight $w = \frac{N}{N + m}$ (with pseudo-count $m = 5.0$):
   $$\text{Par Score} = w \cdot \tilde{x}_{\text{venue}} + (1 - w) \cdot \tilde{x}_{\text{competition}}$$
4. **Transparent Par Differential Bands ($\pm 6.0$ Runs)**:
   In T20 cricket analytics, competitive par represents approximately one standard over ($\sim \pm 6.0$ runs) around the venue's expected par score. Score tiers are classified directly by the run differential $\Delta_{\text{par}} = \hat{y} - \text{Par Score}$:
   $$T_{\text{low}} = \text{Par Score} - 6.0$$
   $$T_{\text{high}} = \text{Par Score} + 6.0$$
   $$\text{Classification} = \begin{cases}
   \text{Below Par}, & \Delta_{\text{par}} < -6.0 \quad (\hat{y} < T_{\text{low}}) \\
   \text{Par}, & -6.0 \le \Delta_{\text{par}} \le +6.0 \quad (T_{\text{low}} \le \hat{y} \le T_{\text{high}}) \\
   \text{Above Par}, & \Delta_{\text{par}} > +6.0 \quad (\hat{y} > T_{\text{high}})
   \end{cases}$$
5. **Persistence & Fast Lookup**:
   Historical statistics are computed once and stored in `data/stadium_par_scores.csv` (and mirrored to `Datasets/stadium_par_scores.csv`). During inference, statistics are looked up in $O(1)$ from an in-memory cache without recalculating from raw deliveries.
6. **Data Leakage Safeguards**:
   - Calculated strictly using matches with `season <= 2021` (832 eligible training first innings).
   - Unseen venues in test seasons ($2022–2024$) fall back automatically to the `_COMPETITION_AVERAGE_` baseline row ($163.5$ par).
   - Functions also support `before_date` for point-in-time rolling validation without peeking into future matches.

### Output Schema

When queried via `predict_score_with_stadium_par()` or the dashboard:
```json
{
  "predicted_score": 150.5,
  "stadium_par_score": 159.3,
  "difference_from_par": -8.8,
  "classification": "Below Par",
  "short_classification": "Low",
  "venue": "MA Chidambaram Stadium, Chepauk",
  "threshold_low": 153.3,
  "threshold_high": 165.3,
  "sample_size": 65,
  "is_fallback": false,
  "par_margin": 6.0
}
```

*Cricket Context Note:* Stadium par reflects historical scoring conditions for the ground. An "Above Par" classification indicates an above-average score for that venue but does not guarantee victory; similarly, a "Below Par" score does not imply it cannot be defended.

---

## 6. Unsupervised Batter Performance Clustering

The system includes an unsupervised learning module (`src/batter_clustering.py`) that identifies empirical batting archetypes from historical ball-by-ball delivery data without relying on human-assigned player roles or subjective labels.

### Purpose
To uncover natural clusters of batting styles and scoring efficiencies across historical IPL seasons, enabling teams and analysts to distinguish high-volume anchors, aggressive finishers, steady accumulators, and lower-order batters.

### Algorithm & Hyperparameters
- **Algorithm**: K-Means Clustering (`sklearn.cluster.KMeans`).
- **Reproducibility**: Deterministic clustering configured with `random_state=42` and `n_init=10`.
- **Search Range**: Evaluates candidate clusters across $k \in [2, 5]$.

### Feature Engineering & Preprocessing
1. **Cricket Delivery Accounting Rules**:
   - `total_runs`: Runs scored off the bat (`batsman_runs`). Team extras (wides, byes, leg-byes) are excluded.
   - `balls_faced`: Deliveries faced, excluding wide deliveries (`extras_type != 'wides'`). No-balls, byes, and leg-byes count as legal balls faced.
   - `dismissals`: Deliveries where `player_dismissed == player` and `dismissal_kind != 'retired hurt'`. Tactical "retired out" is counted as a dismissal under MCC Law 25.4.3.
2. **Primary Clustering Features**:
   - `batting_average`: $\frac{\text{Total Runs}}{\text{Dismissals}}$ (if dismissals $> 0$, otherwise `np.nan`).
   - `strike_rate`: $\frac{\text{Total Runs}}{\text{Balls Faced}} \times 100.0$.
3. **Filtering & Sample Threshold**:
   - **Minimum Activity Threshold ($\ge 200$ balls faced)**: Eliminates small-sample rate volatility from tailenders and brief appearances.
   - **Undefined Averages**: Players with $0$ dismissals are safely filtered out to prevent infinite/NaN distortions.
   - **Chronological Split**: Fitted strictly on matches through `season <= 2021` (yielding $165$ eligible historical batters), maintaining temporal consistency with supervised models.
4. **Feature Standardization (`StandardScaler`)**:
   Because batting average ($\sim 15–40$) and strike rate ($\sim 100–160$) exist on vastly different numerical scales, unscaled Euclidean distance would be dominated by strike rate variance. Features are scaled to zero mean and unit variance ($z = (x - \mu) / \sigma$).

### Evaluation & Interpretation
- **Silhouette Analysis**: Evaluated using `sklearn.metrics.silhouette_score`:
  - $k=2$: **0.3754** (selected optimal $k$)
  - $k=3$: 0.3361
  - $k=4$: 0.3682
  - $k=5$: 0.3591
- **Empirical Clusters ($k=2$)**:
  - **Cluster 0 (High Impact / Primary Scorers)**: $96$ batters, mean batting avg $30.6$, mean strike rate $135.0$ (e.g., Virat Kohli, Shikhar Dhawan, Rohit Sharma).
  - **Cluster 1 (Lower Order / Traditional Accumulators)**: $69$ batters, mean batting avg $22.8$, mean strike rate $115.7$ (e.g., Parthiv Patel, Murali Vijay, Jacques Kallis).
- **Dashboard Visualizations**:
  - Interactive scatter plot of Batting Average vs. Strike Rate colored by cluster.
  - Cluster centroid summary table with mean/median rate stats and top run-scorers.
  - Searchable player-by-player assignment browser with cluster filtering.

---

## 7. How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ShreeBairagi/IPL.git
cd IPL
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Execute the Test Suite
Run unit, integration, and UI tests across all components:
```powershell
.\.venv\Scripts\pytest -q
```
*Includes tests for preprocessing, 10-over snapshots, regression, classification, chase probability, stadium par scoring, batter clustering, and Streamlit app flows (111 tests passing).*

### 5. Generate / Refresh Stadium Par Scores Artifact
To regenerate the stadium par score table strictly from training data:
```powershell
.\.venv\Scripts\python.exe -m src.par_score
```

### 6. Launch the Streamlit Dashboard
Launch the web interface locally:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard contains 5 functional sections:
1. **Project Overview & Data Integrity**: Dataset metrics, chronological split sizes, and leakage safeguards.
2. **First-Innings Final Score Prediction**: Regression forecasting, universal tertiles, and dynamic stadium par evaluation.
3. **Second-Innings Chase-Win Probability**: Dynamic win-probability estimation with live match state inputs.
4. **Model Performance & Evaluation Audit**: Side-by-side performance comparison tables across all tasks and feature sets.
5. **Batter Performance Clustering**: Interactive K-Means cluster space scatter plots, archetype summaries, and qualified player directory.

---

## 8. Future Scope & Planned Enhancements

The following roadmap items represent planned future improvements that are **not yet implemented**:

- **Next-Delivery Wicket Probability (Ball-by-Ball Classification)**: Developing a Naive Bayes or gradient-boosted classification model on individual ball-by-ball events to estimate the conditional probability of a wicket falling on the very next delivery based on bowler type, batsman hand, phase, and dot-ball pressure.
- **Live Match Ingestion & In-Play Streaming**: Connecting real-time ball-by-ball API streams to update predicted final scores, win probabilities, and par differentials dynamically ball by ball.
- **Phase-Wise Innings Forecasting**: Decomposing predictions into distinct match phases: Powerplay (overs 0–5), Middle Overs (overs 6–15), and Death Overs (overs 16–19).
- **Multi-Dimensional Player Clustering**: Expanding clustering beyond batting average and strike rate to include bowler metrics (economy rate, bowling strike rate, dot ball percentage) and fielding impact (catch conversion, run-out involvement).
- **Prediction Intervals & Uncertainty Calibration**: Implementing conformal prediction intervals for regression outputs and probability calibration (e.g., Platt scaling or isotonic regression) for chase-win probabilities.
- **Evaluation on Expanded Historical & Future Seasons**: Validating and expanding historical benchmarks against future IPL seasons ($2025+$) with rolling walk-forward cross-validation.
