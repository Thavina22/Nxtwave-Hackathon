# Model Documentation

## Baseline Model

Logistic Regression with:

- Standard scaling
- One-hot encoding
- Balanced class weights
- Maximum iterations: 2000

## Comparison Model

XGBoost Classifier with class-imbalance handling.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Top 10% failure capture
- Confusion matrix

## Results

### Logistic Regression

- Accuracy: 69.2%
- Precision: 31.1%
- Recall: 70.0%
- F1-score: 43.1%
- ROC-AUC: 78.1%
- PR-AUC: 45.7%
- Top 10% capture: 31.67%

### XGBoost

- Accuracy: 78.3%
- Precision: 36.8%
- Recall: 41.7%
- F1-score: 39.1%
- ROC-AUC: 71.5%
- PR-AUC: 40.0%
- Top 10% capture: 26.67%

## Final Model Selection

Logistic Regression was selected.

In preventive maintenance, missing a genuinely vulnerable line can be more serious than scheduling an unnecessary inspection. Therefore, recall and risk-ranking performance were prioritised over accuracy alone.

## Risk Levels

- 0–29: Low
- 30–54: Medium
- 55–74: High
- 75–100: Critical