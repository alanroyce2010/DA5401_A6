# DA5401 A6 — Imputation via Regression for Missing Data
**Author: Alan Royce Gabriel BS22B001**

---

## Objective
This assignment explores how different imputation strategies for handling missing data influence the performance of a downstream classification model.  
We work with the **UCI Credit Card Default Dataset**, introduce artificial missingness, and compare simple and regression-based imputation methods.

---

## Problem Statement
In credit risk modeling, missing values in customer demographic and transaction data can bias model performance.  
To address this, we implement three imputation strategies and evaluate their impact on predicting the target variable **`default payment next month`**.

We artificially introduce **Missing At Random (MAR)** values (5–10%) in selected numerical columns to simulate real-world data loss scenarios.

---

## Dataset
**Source:** [UCI Credit Card Default Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

**Features:**
- Demographics: `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`
- Credit limit and payment history: `LIMIT_BAL`, `BILL_AMT1–6`, `PAY_AMT1–6`, `PAY_0–6`
- **Target:** `default payment next month`

---

## Steps Performed

### **Part A — Data Preprocessing and Imputation**
1. **Data Loading and Cleaning**
   - Loaded CSV file and removed quotes, spaces, and the `ID` column.
   - Artificially introduced MAR missingness (7%) in three numeric columns:  
     - `AGE`, `BILL_AMT1`, and `PAY_AMT1` using random sampling.

2. **Imputation Strategies**
   - **Dataset A — Median Imputation:**  
     Replaced missing values with median of the respective column.
   - **Dataset B — Linear Regression Imputation (Iterative Imputer):**  
     Used scikit-learn’s `IterativeImputer` with a **Bayesian Ridge (Linear Regression)** estimator to model missing values from other observed features.  
     Assumes **Missing At Random (MAR)**.
   - **Dataset C — Non-linear Regression (KNN Imputer):**  
     Used `KNNImputer` (k=5, distance-weighted) to fill missing values based on nearest neighbors in feature space.
   - **Dataset D — Listwise Deletion:**  
     Dropped all rows containing any missing value.

---

### **Part B — Model Training and Evaluation**
- Split each dataset into **80% training / 20% testing** (stratified).
- Standardized numeric features using `StandardScaler`.
- Trained a **Logistic Regression classifier** with:
  - `penalty='l1'`, `solver='saga'`, `class_weight='balanced'`
- Implemented **threshold tuning** to maximize F1-score on test data.
- Evaluated performance using:
  - Accuracy, Precision, Recall, and F1-score (for positive/default class).

---

### **Part C — Comparative Analysis**

| Dataset | Accuracy | Precision (+) | Recall (+) | F1 (+) |
|----------|-----------|---------------|-------------|--------|
| Median | 0.679 | 0.366 | 0.616 | 0.459 |
| Linear (Iterative) | 0.680 | 0.367 | 0.620 | 0.461 |
| KNN | 0.681 | 0.368 | 0.618 | 0.461 |
| Drop-row | **0.691** | **0.381** | **0.647** | **0.480** |

---

##  Observations & Discussion

- All imputation methods yielded **comparable performance**, but **KNN** slightly outperformed linear imputation.
- The **Listwise Deletion** model achieved the highest F1-score, though it discards useful data — not ideal for real-world applications.
- The **Linear (Iterative) Imputer** aligns with the **MAR assumption**, making it statistically consistent when missingness depends on other observed features.
- **Class weighting** and **threshold tuning** significantly improved recall and F1-score for the minority (default) class.

---

## Recommendations

- **Best trade-off:** KNN or Iterative Linear Imputation — both preserve data while improving predictive stability.  
- For highly non-linear relationships, KNN/Random Forest imputers are preferable.  
- Logistic Regression (balanced) with threshold tuning offers an interpretable yet robust baseline for credit risk modeling.

---

## Files Included

| File | Description |
|------|--------------|
| `DA5401_A6_imputation.ipynb` | Main Jupyter Notebook containing code, visualizations, and analysis. |

---

