# Solution Description

LineGuard AI is an explainable predictive-maintenance dashboard for low-tension electricity-distribution lines.

The system analyses asset condition, historical faults, electrical stress and environmental exposure to calculate a relative 30-day maintenance-risk score for each line segment.

## Main Features

1. District-level risk overview
2. Priority maintenance ranking
3. Individual segment exploration
4. Explainable AI risk contributions
5. Live prediction for new segments
6. Storm and electrical-load simulation
7. Model comparison
8. Downloadable inspection reports

## AI Models

Two models were evaluated:

- Logistic Regression
- XGBoost Classifier

Logistic Regression was selected because it achieved higher recall, F1-score, ROC-AUC, PR-AUC and top-risk failure capture.

## Explainability

The application displays which factors increase or reduce each segment's risk score.

Examples include:

- High line age
- Long inspection interval
- Strong wind exposure
- Repeated overloading
- Previous fault history
- Poor visual condition
- Vegetation proximity

## Output

Each segment receives:

- Relative risk score
- Risk category
- Priority rank
- Top contributing factors
- Recommended inspection action