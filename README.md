# Week 3 Internship Task — Clustering Models & Association Rule Mining

This repository contains the Week 3 deliverables for the internship: unsupervised learning applied to
two real-world datasets — **customer segmentation via clustering** and **market basket analysis via
association rule mining**.

## Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── Mall_Customers.csv       # Part 1 — Mall Customer Segmentation dataset (200 rows)
│   └── groceries.csv            # Part 2 — Groceries Market Basket dataset (9,835 transactions)
└── notebooks/
    ├── week2_clustering.ipynb          # Part 1 — Clustering Models
    └── week2_association_rules.ipynb   # Part 2 — Association Rule Mining
```

## Part 1 — Clustering Models (`week2_clustering.ipynb`)

Customer segmentation on the **Mall Customer Segmentation Dataset** (`Age`, `Annual Income (k$)`,
`Spending Score (1-100)`, `Gender`).

Covers:
1. Data loading and structural review (shape, dtypes, missing values, summary stats)
2. Feature selection (`Annual Income`, `Spending Score`, `Age`) and `StandardScaler` scaling
3. K-Means elbow method across k = 1–10
4. Final K-Means model, 2D cluster visualization, and silhouette score
5. Ward-linkage dendrogram and Agglomerative Clustering
6. DBSCAN with `eps`/`min_samples` tuned via a k-distance plot, including noise-point visualization
7. A comparison table of K-Means vs. Hierarchical vs. DBSCAN (assumptions, outlier handling, need to
   pre-specify k, scalability)
8. A written Clustering Summary covering segment count, most interpretable algorithm, business-facing
   segment descriptions, and a hypothesis on the top-spending segment's behavior

## Part 2 — Association Rule Mining (`week2_association_rules.ipynb`)

Market basket analysis on the **Groceries dataset** (9,835 real grocery-store transactions, 169 unique
items).

Covers:
1. Data loading and structural review, including parsing the basket-format CSV into transaction lists
2. One-hot encoding via `mlxtend.preprocessing.TransactionEncoder`
3. Frequent itemset mining with **Apriori** (`min_support = 0.01`, justified in-notebook)
4. Association rules (support, confidence, lift) from the Apriori itemsets
5. Frequent itemsets and rules with **FP-Growth** at the same `min_support`, with a confirmation that
   both algorithms return identical itemsets
6. A runtime comparison between Apriori and FP-Growth
7. A sensitivity analysis varying `min_support` (0.01/0.02/0.05) and `min_confidence` (0.3/0.5/0.7)
8. A bar chart of the top 10 rules by lift
9. A written Rule Mining Summary covering the highest-lift rules' business meaning, why FP-Growth is
   generally faster than Apriori, and which algorithm to carry forward at larger scale

## Datasets

- **Mall Customer Segmentation Dataset** — `data/Mall_Customers.csv`. 200 customer records with
  `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
- **Groceries Market Basket Dataset** — `data/groceries.csv`. 9,835 real transactions from a grocery
  store, one transaction per row in basket format (item count + item slots).

## Setup & Running

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Both notebooks are self-contained and run top-to-bottom without modification (they read datasets via a
relative `../data/` path, so run them from within the `notebooks/` folder or via Jupyter's default
working directory).

## Key Results at a Glance

- **Clustering:** ~5 meaningful customer segments identified consistently by K-Means and Hierarchical
  Clustering (silhouette score ≈ 0.42 for K-Means at k=5); DBSCAN additionally flags outlier customers
  as noise rather than forcing them into a segment.
- **Association Rules:** At `min_support = 0.01`, both Apriori and FP-Growth find the same 333 frequent
  itemsets and 125 rules — but FP-Growth runs several times faster than Apriori on this dataset, and
  that gap widens with scale.
