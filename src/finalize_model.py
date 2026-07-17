from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TARGET_COLUMN = "failure_within_30_days"


def load_dataset() -> pd.DataFrame:
    """Load the complete synthetic LT-line dataset."""

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "lt_line_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def assign_risk_level(score: float) -> str:
    """Convert the model score into a maintenance-priority tier."""

    if score >= 75:
        return "Critical"

    if score >= 55:
        return "High"

    if score >= 30:
        return "Medium"

    return "Low"


def assign_recommendation(risk_level: str) -> str:
    """Provide an inspection recommendation for each risk tier."""

    recommendations = {
        "Critical": "Inspect within 48 hours",
        "High": "Inspect within 7 days",
        "Medium": "Include in the next scheduled inspection",
        "Low": "Continue routine monitoring",
    }

    return recommendations[risk_level]


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build the final preprocessing and prediction pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
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
        if column not in [
            TARGET_COLUMN,
            "segment_id",
        ]
    ]

    categorical_features = [
        "zone",
        "conductor_material",
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    X = dataset[feature_columns]
    y = dataset[TARGET_COLUMN]

    final_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("\nTraining final LineGuard model...")
    final_pipeline.fit(X, y)

    risk_values = final_pipeline.predict_proba(X)[:, 1]

    scored_segments = dataset.copy()

    # Since the model uses balanced class weights and synthetic data,
    # present this as a relative risk score rather than an exact
    # real-world probability.
    scored_segments["risk_score"] = (
        risk_values * 100
    ).round(1)

    scored_segments["risk_level"] = scored_segments[
        "risk_score"
    ].apply(assign_risk_level)

    scored_segments["recommendation"] = scored_segments[
        "risk_level"
    ].apply(assign_recommendation)

    scored_segments = scored_segments.sort_values(
        by="risk_score",
        ascending=False,
    ).reset_index(drop=True)

    scored_segments["priority_rank"] = (
        scored_segments.index + 1
    )

    project_root = Path(__file__).resolve().parents[1]
    models_directory = project_root / "models"
    data_directory = project_root / "data"

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_directory
        / "final_lineguard_model.joblib"
    )

    scores_path = (
        data_directory
        / "segment_risk_scores.csv"
    )

    metadata_path = (
        models_directory
        / "final_model_metadata.json"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    scored_segments.to_csv(
        scores_path,
        index=False,
    )

    risk_distribution = (
        scored_segments["risk_level"]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "project": "LineGuard AI",
        "selected_model": "Logistic Regression",
        "selection_reason": (
            "Higher recall, F1-score, ROC-AUC, PR-AUC "
            "and top-risk failure capture than XGBoost."
        ),
        "training_records": len(dataset),
        "feature_count": len(feature_columns),
        "target": TARGET_COLUMN,
        "score_description": (
            "Relative model-derived risk score from 0 to 100. "
            "It is not a calibrated real-world failure probability."
        ),
        "risk_thresholds": {
            "Low": "0–29.9",
            "Medium": "30–54.9",
            "High": "55–74.9",
            "Critical": "75–100",
        },
        "risk_distribution": risk_distribution,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print("\nFinal model completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Risk scores saved to: {scores_path}")
    print(f"Metadata saved to: {metadata_path}")

    print("\nRisk-level distribution:")
    print(
        scored_segments["risk_level"].value_counts()
    )

    print("\nTop 10 priority segments:")
    print(
        scored_segments[
            [
                "priority_rank",
                "segment_id",
                "zone",
                "risk_score",
                "risk_level",
                "recommendation",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()