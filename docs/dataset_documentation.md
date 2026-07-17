# Dataset Documentation

## Dataset Type

Physically informed synthetic dataset.

## Reason for Synthetic Data

Actual utility asset, inspection and failure records are confidential and were unavailable during the hackathon.

The prototype therefore uses a simulated dataset designed to represent realistic relationships between line condition, electrical stress, weather exposure and maintenance risk.

## Number of Records

1,800 LT-line segments.

## Target Variable

failure_within_30_days

- 0: No simulated failure or emergency-maintenance requirement
- 1: Simulated failure or emergency-maintenance requirement

## Main Features

- line_age_years
- conductor_material
- span_length_m
- visual_condition_score
- vegetation_proximity_score
- past_fault_count
- months_since_last_inspection
- average_load_percentage
- overload_hours_last_month
- max_wind_speed_kmph
- heavy_rainfall_days

## Class Distribution

- No failure: 1,499 records
- Failure: 301 records
- Failure rate: 16.72%

## Important Disclaimer

The dataset does not represent actual electricity-board infrastructure. Real-world deployment requires retraining and validation using verified utility records.