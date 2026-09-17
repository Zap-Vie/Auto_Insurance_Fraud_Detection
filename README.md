# Auto Insurance Fraud Detection

End-to-end classification pipeline for predicting fraudulent automobile insurance claims using the [Auto Insurance Claims Data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data) from Kaggle.

---

## Project Overview
- **Dataset**: `insurance_claims.csv` (1,000 samples, 39 features after cleaning) downloaded via `kagglehub`.
- **Target Variable**: `fraud_reported` (`Y` $\rightarrow$ 1, `N` $\rightarrow$ 0), showing moderate class imbalance (~24.7% positive class).
- **Primary Goal**: Detect fraudulent claims effectively by prioritizing **Recall** and **F1-Score** alongside **ROC-AUC** and **PR-AUC**.

---

## 🛠️ Data Preprocessing & Feature Engineering
1. **Dropped Features**:
   - High-cardinality/identifier columns (`policy_number`, `insured_zip`, `incident_location`).
   - Empty column `_c39` (100% missing values).
2. **Datetime Decomposition**:
   - `policy_bind_date` $\rightarrow$ `year`, `month`, `day`, `dayofweek`.
   - `incident_date` $\rightarrow$ `year`, `month`, `day`, `dayofweek`.
3. **Missing Value Handling**:
   - Imputed `authorities_contacted` missing values with `"None"`.
4. **Encoding & Scaling (`ColumnTransformer`)**:
   - **Numerical Features**: Scaled via `StandardScaler` (e.g., `total_claim_amount`, `injury_claim`, `vehicle_claim`, `property_claim`, `capital-gains`, `capital-loss`, etc.).
   - **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` (e.g., `policy_state`, `policy_csl`, `incident_type`, `incident_severity`, `insured_hobbies`, etc.).

---

## Modeling & Hyperparameter Tuning
Models were trained with an **80/20 stratified train-test split** (`random_state=42`) and tuned using 5-fold cross-validation optimizing for `f1`:
- **K-Nearest Neighbors (KNN)**: Tuned `n_neighbors`, `weights`, `metric` via `GridSearchCV`.
- **Logistic Regression**: Tuned `C`, `penalty` (L1/L2), and `solver` (`liblinear`, `lbfgs`, `saga`) via `GridSearchCV`.
- **Gaussian Naive Bayes**: Tuned `var_smoothing` via `GridSearchCV`.
- **Decision Tree Classifier**: Tuned `max_depth`, `criterion`, `min_samples_split`, and `min_samples_leaf` via `GridSearchCV`.
- **Random Forest Classifier**: Tuned `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf` via `RandomizedSearchCV`.

---

## 📊 Results & Performance Comparison

| Rank | Model | Accuracy | F1-Score | Recall | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Tuned Random Forest** | **0.840** | **0.750** | **0.873** | **0.844** | **0.656** |
| 2 | **Tuned Logistic Regression** | 0.835 | 0.744 | **0.873** | 0.831 | 0.542 |
| 3 | **Tuned Decision Tree** | 0.805 | 0.683 | 0.764 | 0.814 | 0.604 |
| 4 | Logistic Regression (Baseline) | 0.760 | 0.579 | 0.600 | 0.800 | 0.518 |
| 5 | Decision Tree (Baseline) | 0.735 | 0.505 | 0.491 | 0.659 | 0.395 |
| 6 | Naive Bayes (Baseline / Tuned) | 0.610 | 0.435 | 0.545 | 0.685 | 0.455 |
| 7 | Tuned KNN | 0.685 | 0.323 | 0.273 | 0.576 | 0.340 |
| 8 | Random Forest (Baseline) | 0.720 | 0.300 | 0.218 | 0.822 | 0.528 |
| 9 | KNN (Baseline) | 0.680 | 0.179 | 0.127 | 0.537 | 0.289 |

---

## 🔍 Key Feature Drivers
Across Tree-based models and Logistic Regression, the strongest indicators of fraud were:
1. `incident_severity_Major Damage`: Consistently ranked as the highest positive predictor of fraudulent reports.
2. Specific insured hobbies, notably `insured_hobbies_chess` and `insured_hobbies_cross-fit`.
3. Financial claim features: `property_claim`, `umbrella_limit`, and `policy_annual_premium`
