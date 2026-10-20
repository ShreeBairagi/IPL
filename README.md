# IPL Match Intelligence Lab

An end-to-end, leak-free machine learning and predictive analytics system for the Indian Premier League (IPL), built on ball-by-ball delivery data from the 2008 through 2024 seasons.

The system evaluates mid-innings match states at exactly 10 completed overs (overs 0–9) to forecast first-innings final scores, classify score tiers, estimate second-innings chase-win probabilities, and cluster historical batter profiles using unsupervised learning.

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
├── 4. Unsupervised Batter Clustering (src/batter_clustering.py)
│     └── K-Means clustering (k=4) on historical players (seasons <= 2021) based on scoring efficiency.
├── 5. Feature & Model Ablation Audits
│     ├── Toss Feature Impact Evaluation: Baseline vs. toss-enhanced comparison across all tasks.
│     └── XGBoost Integration (XGBRegressor & MulticlassXGBClassifier): Standardized benchmark against existing models.
└── 6. Interactive Streamlit Dashboard (app.py & src/dashboard_utils.py)
      └── 4-section interactive analytics interface with input validation and live model inference.
```

---

## 2. Chronological Partitioning & Leakage Prevention

To ensure strict real-world generalization and prevent temporal data leakage:
- **Training Set**: Matches from IPL seasons **2008 through 2021** (seasons $\le 2021$).
- **Held-Out Test Set**: Matches from IPL seasons **2022 through 2024** (seasons $\ge 2022$).
- **Partition Integrity**:
  - **Zero match overlap** between training and test sets.
  - All preprocessing transformers (`StandardScaler`, `OneHotEncoder`) are fitted strictly on training data inside scikit-learn `Pipeline` objects.
  - Unseen franchise teams (e.g., Gujarat Titans, Lucknow Super Giants) and venues in 2022+ are handled safely via `handle_unknown="ignore"`.
  - Target variables (`final_score`, `target_achieved`), post-match metadata (`winner`, `result`), and tracking IDs (`match_id`, `season`) are strictly excluded from feature matrices $X$.
  - Score classification thresholds ($152.0$ and $175.0$ runs) are calculated exclusively from training set tertiles ($33.33\%$ and $66.67\%$).

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
| **Batter Clustering** | K-Means ($k=2$ to $5$, optimal $k=4$, `n_init=10`, `random_state=42`) | Silhouette Score, Calinski-Harabasz, Davies-Bouldin |

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

## 5. How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ShreeBairagi/IPL.git
cd IPL
```

### 2. Create a virtual environment

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

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

The dashboard may take a little time to initialize on its first launch.

### Requirements

- Python 3.11
- The datasets included in the repository
- Dependencies listed in `requirements.txt`

### Environment Setup
Ensure the local Python environment (`.venv`) is activated.

### 1. Execute the Test Suite
Run all unit, integration, and dashboard tests across all modules:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 2. Launch the Streamlit Dashboard
Launch the interactive web application locally:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard contains 4 functional sections:
1. **Project Overview & Data Integrity**: Dataset metrics, split sizes, and leakage safeguards.
2. **First-Innings Final Score Prediction & Classification**: Interactive scenario forecasting.
3. **Second-Innings Chase-Win Probability**: Dynamic win-probability estimator.
4. **Model Performance & Evaluation Audit**: Side-by-side performance comparison tables across all tasks and feature sets.
