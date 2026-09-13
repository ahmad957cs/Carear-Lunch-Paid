
# Career Launchpad Program — AI / Data Science Internship

Portfolio of internship deliverables covering unsupervised learning, market basket analysis,
regression-based predictive modeling with dashboard deployment, classical time-series forecasting,
and deep learning (ANN/CNN/RNN/YOLO). Each task follows the full data-science lifecycle: EDA →
feature engineering → modeling → evaluation → reporting, and every notebook in this repository
runs end-to-end and produces the metrics and plots referenced below.

**Intern:** Ahmad Gul — BS Computer Science(Ai), UET Mardan | AI/ML Engineering focus

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Task 1 — Customer Segmentation (Clustering)](#task-1--customer-segmentation-clustering)
3. [Task 2 — Market Basket Analysis (Association Rule Mining)](#task-2--market-basket-analysis-association-rule-mining)
4. [Task 3 — Steel Industry Energy Prediction & Dashboard](#task-3--steel-industry-energy-prediction--dashboard)
5. [Task 4 — Time-Series Forecasting & Deep Learning](#task-4--time-series-forecasting--deep-learning)
6. [Tech Stack](#tech-stack)
7. [Setup & Installation](#setup--installation)
8. [How to Run](#how-to-run)
9. [Results Summary](#results-summary)
10. [Notes & Known Limitations](#notes--known-limitations)

---

## Repository Structure

```
.
├── README.md                          — this file
├── requirements.txt                   — consolidated package versions
│
├── Mall_Customers.csv                 — Task 1 dataset
├── week2_clustering.ipynb             — Task 1 notebook
│
├── groceries.csv                      — Task 2 dataset
├── week2_association_rules.ipynb      — Task 2 notebook
│
├── Steel_industry_data.csv            — Task 3 raw dataset
├── Steel_Energy_Engineered.csv        — Task 3 dataset after feature engineering
├── Steel_EDA.ipynb                    — Task 3 EDA + feature engineering + modeling notebook
├── app.py                             — Task 3 Streamlit dashboard entry point
├── data_loader.py                     — Task 3 dashboard module (data ingestion) — WIP
├── eda.py                             — Task 3 dashboard module (EDA page) — WIP
├── feature_engineering.py             — Task 3 dashboard module (feature pipeline) — WIP
├── modeling.py                        — Task 3 dashboard module (model training) — WIP
├── evaluation.py                      — Task 3 dashboard module (metrics/plots) — WIP
├── visualization.py                   — Task 3 dashboard module (shared charts) — WIP
├── main.py                            — Task 3 dashboard orchestrator — WIP
│
├── time_series_ARIMA_Prophet.ipynb    — Task 4A: ARIMA & Prophet forecasting
├── analysis.md                        — Task 4A written analysis
├── part_a_comparison_table.csv        — Task 4A ARIMA vs Prophet metrics
│
├── deep_learning_ANN_CNN_RNN.ipynb    — Task 4B: ANN, CNN, RNN
├── yolo_inference.ipynb               — Task 4B: YOLO object detection
├── summary_report.md                  — Task 4B comparative report
├── part_b_summary_table.csv           — Task 4B architecture accuracy comparison
├── part_b_yolo_detections.csv         — Task 4B YOLO detection log
└── report.md                          — reserved / not yet populated
```

---

## Task 1 — Customer Segmentation (Clustering)

**Notebook:** `week2_clustering.ipynb` · **Dataset:** `Mall_Customers.csv` (200 customers: Gender,
Age, Annual Income, Spending Score)

**Objective:** Segment mall customers into actionable marketing groups using unsupervised learning.

**Approach:**
- EDA: distribution checks, missing-value/duplicate audit, gender split
- Feature scaling with `StandardScaler` on Age, Annual Income, and Spending Score
- **K-Means** — cluster count selected via the elbow method and confirmed with silhouette scores
- **Hierarchical Clustering** (Ward linkage) — validated against a dendrogram cut
- **DBSCAN** — `eps` tuned via a k-distance plot, used as an outlier-detection diagnostic
- Side-by-side comparison of all three algorithms' assumptions and outputs

**Key findings:**
- All three methods converge on **5 meaningful customer segments** on the Income–Spending plane
- **K-Means** gave the most business-interpretable result, since the underlying groups are roughly
  spherical and similarly sized — exactly the structure K-Means is designed to find
- **DBSCAN's** main value-add was flagging atypical/outlier customers rather than defining segments
- Segments were mapped to business personas: *Premium/VIP*, *Careful/Reserved*, *Impulsive/
  Aspirational*, *Budget-conscious*, and *Mainstream/Standard*

---

## Task 2 — Market Basket Analysis (Association Rule Mining)

**Notebook:** `week2_association_rules.ipynb` · **Dataset:** `groceries.csv` (9,835 transactions,
169 unique items)

**Objective:** Discover cross-sell relationships between grocery items using frequent itemset
mining.

**Approach:**
- Parsed variable-length transaction rows and one-hot encoded them with `TransactionEncoder`
- Mined frequent itemsets with both **Apriori** and **FP-Growth** (`min_support = 0.01`)
- Generated association rules (`min_confidence = 0.3`) and ranked them by **lift**
- Benchmarked runtime of Apriori vs. FP-Growth across repeated runs
- Ran a sensitivity analysis over a grid of `min_support` / `min_confidence` values

**Key findings:**
- Apriori and FP-Growth provably return the **identical set of frequent itemsets** — confirmed
  empirically — but **FP-Growth is faster and scales better**, since it mines a compact FP-tree
  after two database passes instead of Apriori's repeated candidate-generate-and-test scans
- High-support items like `whole milk` produce high-confidence but **low-lift** rules (they're just
  common), whereas the highest-lift rules pair niche/specialty items — the genuinely useful
  cross-sell signals for shelf placement or bundled promotions
- **FP-Growth** is the recommended algorithm to carry forward at production/retail scale

---

## Task 3 — Steel Industry Energy Prediction & Dashboard

**Notebook:** `Steel_EDA.ipynb` · **Dataset:** `Steel_industry_data.csv` (35,040 fifteen-minute
readings: energy usage, reactive power, CO₂, power factor, load type, day/week status)

**Objective:** Predict industrial energy consumption (`Usage_kWh`) from operating conditions and
timestamp-derived features, and expose the pipeline through an interactive dashboard.

**Approach:**
- **EDA:** trend/outlier analysis (IQR + boxplots), correlation heatmap, average usage by load
  type and hour of day, missing-value audit and visualization
- **Feature engineering:** extracted `Hour` and `Day_of_Week` from the timestamp, imputed
  `Power_Factor_Ratio` with the median, one-hot encoded categorical fields, exported the engineered
  table to `Steel_Energy_Engineered.csv`
- **Modeling:** trained and compared multiple regressors (Linear Regression, Ridge, and additional
  models), evaluated with an 80/20 train-test split plus cross-validation, and selected the
  best-performing model by RMSE
- **Deployment:** a multi-page **Streamlit dashboard** (`app.py`) scaffolded with Home, Dataset,
  EDA, Feature Engineering, Model Training, Evaluation, Prediction, and About pages

**Status note:** the supporting modules (`data_loader.py`, `eda.py`, `feature_engineering.py`,
`modeling.py`, `evaluation.py`, `visualization.py`, `main.py`) are scaffolded as placeholders for
refactoring the notebook logic into a modular dashboard backend — see
[Notes & Known Limitations](#notes--known-limitations).

---

## Task 4 — Time-Series Forecasting & Deep Learning

### 4A — ARIMA & Prophet (`time_series_ARIMA_Prophet.ipynb`)

**Dataset:** Mauna Loa atmospheric CO₂ concentration, resampled to 527 monthly observations
(1958–2001).

- Stationarity testing (ADF), ACF/PACF analysis, and differencing
- `auto_arima`-selected seasonal **SARIMAX** model with confidence intervals
- **Prophet** model with yearly seasonality
- Head-to-head comparison on a 20% held-out test window (see `part_a_comparison_table.csv` and
  `analysis.md`)

| Model            | MAE    | RMSE   | MAPE (%) |
|------------------|--------|--------|----------|
| ARIMA (SARIMAX)  | 1.3873 | 1.6583 | 0.3784   |
| **Prophet**      | **0.6968** | **0.8634** | **0.1927** |

Prophet edged out ARIMA on every metric, likely because its flexible piecewise-linear trend
captures the CO₂ series' gentle acceleration better than ARIMA's fixed autoregressive structure —
full discussion in `analysis.md`.

### 4B — Deep Learning: ANN, CNN, RNN, YOLO

**Notebooks:** `deep_learning_ANN_CNN_RNN.ipynb`, `yolo_inference.ipynb`

| Architecture | Task                                              | Test Accuracy |
|--------------|----------------------------------------------------|:---:|
| ANN          | Tabular binary classification (breast cancer)       | 0.9561 |
| CNN          | Image classification, 10 classes (digits)           | 0.9861 |
| SimpleRNN    | Sequence classification (synthetic sentiment)       | 0.9988 |
| LSTM         | Sequence classification (synthetic sentiment)       | 1.0000 |

- **YOLO** (YOLOv4-tiny via OpenCV DNN) ran object detection across 8 sample images, correctly
  identifying people, animals, and vehicles with CPU inference times of **93–463 ms/image**
  (`part_b_yolo_detections.csv`)
- Full architecture-by-architecture discussion of complexity, training time, and use-case fit is in
  `summary_report.md`

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas, numpy |
| Classical ML | scikit-learn, mlxtend (Apriori/FP-Growth), scipy |
| Time series | statsmodels, pmdarima (`auto_arima`), Prophet |
| Deep learning | TensorFlow / Keras |
| Computer vision | OpenCV (`dnn` module), YOLOv4-tiny |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |
| Environment | Jupyter Notebook |

---

## Setup & Installation

```bash
git clone https://github.com/ahmad957cs/Carear-Lunch-Paid.git
cd Carear-Lunch-Paid

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> `mlxtend` (Apriori/FP-Growth) and `streamlit` are used by Task 2 and Task 3 respectively but are
> not yet pinned in `requirements.txt` — add them with `pip install mlxtend streamlit` until the
> file is updated (see [Notes & Known Limitations](#notes--known-limitations)).

## How to Run

**Notebooks** (Tasks 1, 2, 4):
```bash
jupyter notebook
# open any .ipynb file and run all cells top-to-bottom
```

**Steel Industry dashboard** (Task 3):
```bash
streamlit run app.py
```

---

## Results Summary

| Task | Best result |
|---|---|
| Customer Segmentation | 5 well-separated segments; K-Means most interpretable |
| Market Basket Analysis | FP-Growth ≈ Apriori accuracy, faster and more scalable |
| Steel Energy Prediction | Best regressor selected by RMSE after cross-validation (see `Steel_EDA.ipynb`) |
| ARIMA vs Prophet | Prophet lower error on all metrics (MAPE 0.19% vs 0.38%) |
| ANN / CNN / RNN | 95.6% / 98.6% / up to 100% test accuracy |
| YOLO detection | Correct multi-class detections, <0.5s/image on CPU |

---

## Notes & Known Limitations

- **Relative data paths:** `Steel_EDA.ipynb`, `week2_clustering.ipynb`, and
  `week2_association_rules.ipynb` read from `../data/...`, which assumes the notebook lives one
  level below a shared `data/` folder. In this repository's current flat layout, either run those
  notebooks from a `notebooks/` subfolder with the CSVs moved into a sibling `data/` folder, or
  update the paths to point at the repository root.
- **Dashboard modules are placeholders:** `data_loader.py`, `eda.py`, `feature_engineering.py`,
  `modeling.py`, `evaluation.py`, `visualization.py`, and `main.py` are currently empty stubs.
  `app.py` defines the page navigation shell; the logic from `Steel_EDA.ipynb` still needs to be
  refactored into these modules for the dashboard to be fully functional.
- **`report.md`** is reserved but not yet populated.
- **Task 4 dataset substitutions:** the Deep Learning and YOLO notebooks were originally built in a
  network-restricted sandbox that could not reach Kaggle, Yahoo Finance, or the servers Keras's
  built-in loaders use. They therefore substitute equivalent offline datasets (`scikit-learn`'s
  breast-cancer and digits datasets, a synthetic sequence-classification set, and YOLOv4-tiny via
  OpenCV instead of `ultralytics` YOLOv8n). This is flagged inline in each notebook, along with the
  one-line change needed to restore the originally suggested data source on a machine with full
  internet access.
- **`requirements.txt`** currently lists only the Task 4 (time-series + deep learning) dependency
  set; `mlxtend`, `streamlit`, and `seaborn` should be added to make it cover the whole repository.
