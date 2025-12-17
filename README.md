Full Project: https://jimena-judith.github.io/Loan-Default-Prediction---A-Binary-Classification-Approach/

# Loan-Default-Prediction: A Binary Classification Approach

## Project Overview
This project predicts **loan default** using borrower, loan, and property features, along with engineered nonlinear interactions. A central focus is a **leakage-free preprocessing pipeline** to ensure realistic model evaluation.

## Methodology
- **Data Preprocessing (Leakage-Free):**
    - Imputation of missing values using structure-aware methods.
    - Outlier handling via winsorization (0.5–99.5 percentiles) or domain-informed clipping (e.g., DTI, LTV).
    - Categorical encoding:
        - Binary → Label Encoding
        - Multi-class → One-Hot Encoding
    - Scaling applied only on training data, then applied to test set.
- **Feature Engineering:**
    - Engineered interactions and ratios between loan, income, and property.
    - Risk-relevant metrics: affordability ratios, LTV, DTI.

## Models Evaluated
- **Linear Models:** Logistic Regression (L1, L2, Ridge)
- **Tree-Based:** Decision Tree, Random Forest
- **Gradient Boosting:** XGBoost
- Hyperparameters tuned via cross-validation; class imbalance handled with class weights.

## Results (Leak-Free Test Set)
| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|---------|--------|----|---------|
| **XGBoost** | 0.87 | **0.72** | **0.73** | **0.89** |
| Random Forest | 0.88 | 0.58 | 0.71 | 0.88 |
| Decision Tree | 0.86 | 0.64 | 0.69 | 0.84 |
| Logistic Regression | 0.83 | 0.64 | 0.65 | 0.84 |

### Insights
- Linear models provide stable baselines but cannot fully capture nonlinear interactions.
- Decision Tree captures some interactions but is unstable.
- Random Forest improves precision but sacrifices recall.
- **XGBoost balances recall and overall discrimination**, effectively modeling nonlinear relationships and feature interactions.

## Conclusion
- Leakage-free pipelines are essential for realistic evaluation.
- Gradient boosting is best suited for complex credit-risk data.
- Feature engineering focusing on borrower affordability, collateral, and loan structure yields stable predictive signals.
- Future work: alternative imbalance handling (SMOTE, thresholds), additional boosting methods, interpretability with SHAP.
