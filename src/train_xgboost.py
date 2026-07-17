from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TARGET_COLUMN = "failure_within_30_days"


def load_dataset() -> pd.DataFrame:
    """Load the synthetic LT-line dataset."""

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "lt_line_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run generate_data.py first."
        )

    return pd.read_csv(dataset_path)


def calculate_top_risk_capture(
    y_true: pd.Series,
    probabilities: np.ndarray,
    percentage: float = 0.10,
) -> float:
    """Measure failures captured in the top-risk percentage."""

    results = pd.DataFrame(
        {
            "actual": y_true.reset_index(drop=True),
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)

    top_count = max(1, int(len(results) * percentage))
    total_failures = results["actual"].sum()

    if total_failures == 0:
        return 0.0

    captured_failures = results.head(top_count)["actual"].sum()

    return float(captured_failures / total_failures)


def assign_risk_level(score: float) -> str:
    """Convert risk score into a maintenance priority level."""

    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    """Create preprocessing and XGBoost pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        sparse_threshold=0,
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    dataset = load_dataset()

    feature_columns = [
        column
        for column in dataset.columns
        if column not in [TARGET_COLUMN, "segment_id"]
    ]

    X = dataset[feature_columns]
    y = dataset[TARGET_COLUMN]

    categorical_features = [
        "zone",
        "conductor_material",
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("\nDataset loaded successfully.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Training failures: {positive_count}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
    )

    print("\nTraining XGBoost model...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Initially use the normal 0.50 threshold.
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    top_10_capture = calculate_top_risk_capture(
        y_test,
        probabilities,
        percentage=0.10,
    )

    top_20_capture = calculate_top_risk_capture(
        y_test,
        probabilities,
        percentage=0.20,
    )

    matrix = confusion_matrix(y_test, predictions)

    print("\n--- XGBoost Model Results ---")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}")
    print(
        "Top 10% risk capture:",
        f"{top_10_capture * 100:.2f}%",
    )
    print(
        "Top 20% risk capture:",
        f"{top_20_capture * 100:.2f}%",
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
            zero_division=0,
        )
    )

    project_root = Path(__file__).resolve().parents[1]
    models_directory = project_root / "models"
    data_directory = project_root / "data"

    models_directory.mkdir(parents=True, exist_ok=True)

    model_path = models_directory / "xgboost_model.joblib"
    joblib.dump(pipeline, model_path)

    metrics = {
        "model": "XGBoost Classifier",
        "threshold": threshold,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "top_10_percent_risk_capture": round(
            float(top_10_capture),
            4,
        ),
        "top_20_percent_risk_capture": round(
            float(top_20_capture),
            4,
        ),
        "confusion_matrix": matrix.tolist(),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    metrics_path = models_directory / "xgboost_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    test_results = X_test.copy()
    test_results["actual_failure"] = y_test.values
    test_results["predicted_failure"] = predictions
    test_results["failure_probability"] = probabilities.round(4)
    test_results["risk_score"] = (
        probabilities * 100
    ).round(1)
    test_results["risk_level"] = test_results[
        "risk_score"
    ].apply(assign_risk_level)

    predictions_path = (
        data_directory / "xgboost_test_predictions.csv"
    )

    test_results.to_csv(
        predictions_path,
        index=False,
    )

    print("\nFiles saved successfully:")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()