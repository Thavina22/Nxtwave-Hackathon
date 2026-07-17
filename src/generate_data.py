from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility settings
RANDOM_SEED = 42
NUMBER_OF_SEGMENTS = 1800


# Fictional zone centres used only for dashboard visualisation.
# These coordinates do not represent real electricity-board assets.
ZONE_CENTRES = {
    "North": (13.08, 80.16),
    "South": (12.72, 80.02),
    "East": (12.93, 80.21),
    "West": (12.94, 79.93),
    "Central": (12.91, 80.08),
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Convert a raw risk value into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-value))


def generate_dataset(
    number_of_segments: int = NUMBER_OF_SEGMENTS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a physically informed synthetic LT line dataset.

    The dataset is not real utility data. It simulates realistic
    relationships between asset condition, electrical stress,
    environmental exposure and 30-day failure risk.
    """

    rng = np.random.default_rng(random_seed)

    # ---------------------------------------------------------
    # 1. Location and zone information
    # ---------------------------------------------------------

    zones = rng.choice(
        list(ZONE_CENTRES.keys()),
        size=number_of_segments,
        p=[0.18, 0.22, 0.18, 0.22, 0.20],
    )

    latitudes = np.array(
        [ZONE_CENTRES[zone][0] for zone in zones]
    ) + rng.normal(0, 0.035, number_of_segments)

    longitudes = np.array(
        [ZONE_CENTRES[zone][1] for zone in zones]
    ) + rng.normal(0, 0.035, number_of_segments)

    # ---------------------------------------------------------
    # 2. Asset characteristics
    # ---------------------------------------------------------

    line_age_years = rng.integers(
        low=1,
        high=36,
        size=number_of_segments,
    )

    conductor_material = rng.choice(
        ["AAC", "ACSR", "Copper"],
        size=number_of_segments,
        p=[0.45, 0.40, 0.15],
    )

    span_length_m = np.clip(
        rng.normal(35, 10, number_of_segments),
        15,
        70,
    )

    # 1 = very good condition
    # 10 = very poor condition
    visual_condition_score = rng.integers(
        1,
        11,
        number_of_segments,
    )

    # 0 = vegetation far away
    # 10 = vegetation extremely close
    vegetation_proximity_score = rng.integers(
        0,
        11,
        number_of_segments,
    )

    # ---------------------------------------------------------
    # 3. Fault and inspection history
    # ---------------------------------------------------------

    past_fault_count = np.clip(
        rng.poisson(1.1, number_of_segments),
        0,
        7,
    )

    months_since_last_inspection = rng.integers(
        1,
        37,
        number_of_segments,
    )

    # ---------------------------------------------------------
    # 4. Electrical stress
    # ---------------------------------------------------------

    average_load_percentage = np.clip(
        rng.normal(78, 18, number_of_segments),
        35,
        130,
    )

    overload_hours_last_month = np.clip(
        rng.gamma(2, 10, number_of_segments)
        + (
            average_load_percentage > 95
        ) * rng.uniform(10, 40, number_of_segments),
        0,
        100,
    )

    # ---------------------------------------------------------
    # 5. Environmental exposure
    # ---------------------------------------------------------

    max_wind_speed_kmph = np.clip(
        rng.normal(42, 16, number_of_segments),
        10,
        100,
    )

    heavy_rainfall_days = np.clip(
        rng.poisson(3.5, number_of_segments),
        0,
        12,
    )

    # ---------------------------------------------------------
    # 6. Hidden synthetic risk-generation process
    # ---------------------------------------------------------

    material_risk = np.select(
        [
            conductor_material == "AAC",
            conductor_material == "ACSR",
            conductor_material == "Copper",
        ],
        [
            0.35,
            0.15,
            -0.20,
        ],
    )

    raw_risk = (
        -6.5
        + 0.065 * line_age_years
        + 0.23 * past_fault_count
        + 0.032 * months_since_last_inspection
        + 0.11 * visual_condition_score
        + 0.08 * vegetation_proximity_score
        + 0.012 * np.maximum(
            average_load_percentage - 75,
            0,
        )
        + 0.012 * overload_hours_last_month
        + 0.025 * np.maximum(
            max_wind_speed_kmph - 35,
            0,
        )
        + 0.11 * heavy_rainfall_days
        + material_risk
    )

    # Interaction effects make the problem nonlinear.
    raw_risk += (
        0.70
        * (
            (line_age_years > 20)
            & (max_wind_speed_kmph > 60)
        )
    )

    raw_risk += (
        0.60
        * (
            (vegetation_proximity_score > 7)
            & (max_wind_speed_kmph > 50)
        )
    )

    raw_risk += (
        0.80
        * (
            (past_fault_count >= 3)
            & (months_since_last_inspection > 18)
        )
    )

    # Real infrastructure behaviour contains uncertainty.
    raw_risk += rng.normal(
        0,
        0.55,
        number_of_segments,
    )

    failure_probability = sigmoid(raw_risk)

    failure_within_30_days = rng.binomial(
        1,
        failure_probability,
    )

    # ---------------------------------------------------------
    # 7. Create final dataset
    # ---------------------------------------------------------

    dataset = pd.DataFrame(
        {
            "segment_id": [
                f"LT-{index:04d}"
                for index in range(
                    1,
                    number_of_segments + 1,
                )
            ],
            "zone": zones,
            "latitude": latitudes.round(6),
            "longitude": longitudes.round(6),
            "line_age_years": line_age_years,
            "conductor_material": conductor_material,
            "span_length_m": span_length_m.round(1),
            "visual_condition_score": visual_condition_score,
            "vegetation_proximity_score": (
                vegetation_proximity_score
            ),
            "past_fault_count": past_fault_count,
            "months_since_last_inspection": (
                months_since_last_inspection
            ),
            "average_load_percentage": (
                average_load_percentage.round(1)
            ),
            "overload_hours_last_month": (
                overload_hours_last_month.round(1)
            ),
            "max_wind_speed_kmph": (
                max_wind_speed_kmph.round(1)
            ),
            "heavy_rainfall_days": heavy_rainfall_days,
            "failure_within_30_days": (
                failure_within_30_days
            ),
        }
    )

    return dataset


def save_dataset(dataset: pd.DataFrame) -> Path:
    """Save the generated dataset inside the project's data folder."""

    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

    output_path = data_directory / "lt_line_dataset.csv"

    dataset.to_csv(
        output_path,
        index=False,
    )

    return output_path


def main() -> None:
    dataset = generate_dataset()
    output_path = save_dataset(dataset)

    failure_counts = dataset[
        "failure_within_30_days"
    ].value_counts().sort_index()

    failure_percentage = (
        dataset["failure_within_30_days"].mean() * 100
    )

    print("\nLineGuard AI synthetic dataset created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Rows: {dataset.shape[0]}")
    print(f"Columns: {dataset.shape[1]}")

    print("\nTarget distribution:")
    print(f"No failure within 30 days: {failure_counts.get(0, 0)}")
    print(f"Failure within 30 days: {failure_counts.get(1, 0)}")
    print(f"Failure percentage: {failure_percentage:.2f}%")

    print("\nFirst five records:")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()s