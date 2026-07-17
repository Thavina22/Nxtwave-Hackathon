# LineGuard AI

## Explainable Predictive Risk Scoring for Preventive LT-Line Maintenance

LineGuard AI is an AI-powered preventive-maintenance platform that identifies which low-tension electricity-line segments may require urgent inspection.

It analyses asset condition, previous faults, electrical loading, inspection history and environmental exposure to generate a relative risk score from 0 to 100.

The goal is to help utility maintenance teams prioritise vulnerable line segments before they become outages or public-safety emergencies.

---


**Idea2Impact 2026 Online Hackathon**

**Selected Theme:** Theme 3 — Crisis Management, HealthTech & Emergency Response

**Relevant Domains:**

- Public Safety
- Emergency Management
- Community Resilience
- Government and Public Administration

---

## One-Line Pitch

> LineGuard AI predicts which LT power-line segments require priority inspection before they become outages or electrocution hazards.

---

## Problem Statement

Low-tension overhead power lines form the final electricity-distribution connection to homes, shops and communities.

In rural and semi-urban areas, these line segments may be affected by:

- Aging infrastructure
- Strong winds and heavy rainfall
- Vegetation contact
- Repeated electrical overloading
- Corrosion and poor physical condition
- Previous faults
- Long intervals between inspections

Utility teams often become aware of vulnerable infrastructure only after:

- A conductor breaks
- An outage occurs
- A public complaint is received
- Emergency maintenance becomes necessary

Maintenance teams may be responsible for thousands of line segments but have limited workers, vehicles and inspection time.

Without a predictive system, maintenance remains periodic, complaint-based or reactive.

---

## Who It Affects

LineGuard AI is designed to support:

- Electricity-distribution utilities
- District electricity boards
- Utility maintenance teams
- Field inspection engineers
- Rural and semi-urban communities
- People and animals near damaged electrical infrastructure

---

## Why Existing Approaches Are Insufficient

Circuit breakers and electrical-protection systems primarily respond after abnormal electrical conditions develop.

They do not provide maintenance teams with a preventive ranking showing which line segments are gradually becoming vulnerable.

Manual inspection of every line segment is:

- Expensive
- Time-consuming
- Labour-intensive
- Difficult during severe weather
- Inefficient when maintenance resources are limited

As a result, relatively healthy segments may be inspected while more vulnerable segments remain unattended.

---

## Proposed Solution

LineGuard AI combines multiple line-segment factors, including:

- Line age
- Conductor material
- Span length
- Visual condition
- Vegetation proximity
- Previous fault count
- Months since inspection
- Average electrical load
- Overload hours
- Maximum wind exposure
- Heavy-rainfall days

The trained model generates:

- Relative risk score from 0 to 100
- Risk category
- District priority rank
- Main risk-increasing factors
- Risk-reducing factors
- Recommended inspection timeline

---

## AI at the Core

Two machine-learning models were trained and evaluated:

1. Logistic Regression
2. XGBoost Classifier

Logistic Regression was selected as the final model because it achieved better safety-focused performance.

For preventive maintenance, missing a genuinely vulnerable segment may be more serious than scheduling an unnecessary inspection.

Therefore, the model was selected mainly using:

- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Top-risk failure capture

The final model uses:

- Standard scaling for numerical features
- One-hot encoding for categorical features
- Balanced class weights
- Probability-based relative risk scoring
- Feature-level prediction explanations

---

## Model Results

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 69.2% |
| Precision | 31.1% |
| Recall | 70.0% |
| F1-score | 43.1% |
| ROC-AUC | 78.1% |
| PR-AUC | 45.7% |
| Top 10% Failure Capture | 31.67% |

### XGBoost

| Metric | Result |
|---|---:|
| Accuracy | 78.3% |
| Precision | 36.8% |
| Recall | 41.7% |
| F1-score | 39.1% |
| ROC-AUC | 71.5% |
| PR-AUC | 40.0% |
| Top 10% Failure Capture | 26.67% |

### Why Logistic Regression Was Selected

Although XGBoost achieved higher accuracy and precision, Logistic Regression detected more actual failure cases.

Logistic Regression correctly detected:

- 42 out of 60 simulated failure cases

XGBoost correctly detected:

- 25 out of 60 simulated failure cases

For a preventive-maintenance system, higher failure recall was prioritised over overall accuracy.

---

## Risk Categories

| Risk Score | Risk Level | Recommended Action |
|---:|---|---|
| 0–29 | Low | Continue routine monitoring |
| 30–54 | Medium | Include in the next scheduled inspection |
| 55–74 | High | Inspect within 7 days |
| 75–100 | Critical | Inspect within 48 hours |

The score is a relative model-derived maintenance priority and not a guaranteed real-world failure probability.

---

## Application Features

### 1. District Overview

Displays:

- Total line segments
- Critical segments
- High-risk segments
- Average district risk
- Number of priority inspections
- Risk-level distribution
- Average risk by zone
- Top maintenance priorities
- Downloadable inspection report

### 2. Segment Explorer

Allows the user to select an individual line segment and view:

- Relative risk score
- Risk category
- District priority rank
- Asset condition
- Electrical stress
- Environmental exposure
- Main AI risk drivers
- Risk-reducing factors
- Segment location
- Maintenance recommendation

### 3. Predict New Segment

Allows maintenance teams to enter new segment details and receive:

- Live AI-generated risk score
- Risk classification
- Recommended action
- Explainable feature contributions

### 4. Storm Impact Simulator

Simulates changes in:

- Wind speed
- Heavy-rainfall days
- Electrical load
- Overload hours

The simulator displays:

- Critical segments before the storm
- Critical segments after the storm
- Newly critical segments
- Newly prioritised segments
- Average-risk increase
- Before-versus-after risk distribution
- Highest-priority segments after the storm
- Downloadable storm inspection report

### 5. Model Performance

Compares Logistic Regression and XGBoost using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Top 10% failure capture
- Confusion matrices

---

## System Workflow

```text
Synthetic LT-Line Dataset
        ↓
Data Cleaning and Feature Engineering
        ↓
Train Logistic Regression and XGBoost
        ↓
Evaluate Safety-Focused Metrics
        ↓
Select Logistic Regression
        ↓
Generate Relative Risk Scores
        ↓
Explain Risk Contributions
        ↓
Display Results in Streamlit Dashboard
        ↓
Generate Maintenance Priority Reports