from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st



st.set_page_config(
    page_title="LineGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "LineGuard AI — Explainable predictive risk scoring "
            "for preventive LT-line maintenance."
        ),
    },
)



st.markdown(
    """
    <style>
    :root {
        --navy-950: #07111F;
        --navy-900: #0F172A;
        --navy-800: #1E293B;
        --blue-700: #1D4ED8;
        --blue-600: #2563EB;
        --blue-100: #DBEAFE;
        --slate-700: #334155;
        --slate-600: #475569;
        --slate-500: #64748B;
        --slate-300: #CBD5E1;
        --slate-200: #E2E8F0;
        --slate-100: #F1F5F9;
        --surface: #FFFFFF;
        --workspace: #F6F8FB;
        --critical: #DC2626;
        --high: #EA580C;
        --medium: #D97706;
        --low: #15803D;
    }

    /* App workspace */
    .stApp {
        background:
            radial-gradient(
                circle at 95% 5%,
                rgba(37, 99, 235, 0.055),
                transparent 24rem
            ),
            var(--workspace);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    /* Typography */
    h1 {
        color: var(--navy-900);
        font-size: 2.05rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.035em;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        color: var(--navy-900);
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h3 {
        color: var(--navy-900);
        font-size: 1.12rem !important;
        font-weight: 700 !important;
    }

    p,
    li {
        color: var(--slate-600);
        line-height: 1.65;
    }

    hr {
        border: 0;
        border-top: 1px solid var(--slate-200);
        margin: 1.7rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07111F 0%,
                #0F172A 52%,
                #111C31 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.15rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        padding: 0.62rem 0.7rem;
        border-radius: 8px;
        transition:
            background 0.18s ease,
            transform 0.18s ease;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: rgba(96, 165, 250, 0.12);
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"]
    div[data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.20);
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.20);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        min-height: 126px;
        padding: 1.05rem 1.1rem;
        background: var(--surface);
        border: 1px solid var(--slate-200);
        border-radius: 12px;
        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.03),
            0 8px 24px rgba(15, 23, 42, 0.045);
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 2px 5px rgba(15, 23, 42, 0.05),
            0 12px 30px rgba(15, 23, 42, 0.075);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--slate-500);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy-900);
        font-size: 1.75rem;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 650;
    }

    /* Forms and inputs */
    div[data-testid="stForm"] {
        padding: 1.4rem;
        background: var(--surface);
        border: 1px solid var(--slate-200);
        border-radius: 14px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.04);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
    }

    div[data-testid="stSlider"] {
        padding-top: 0.25rem;
        padding-bottom: 0.35rem;
    }

    /* Buttons */
    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 42px;
        border-radius: 8px;
        border: 1px solid var(--blue-700);
        background: var(--blue-700);
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: 0.005em;
        box-shadow: 0 3px 8px rgba(29, 78, 216, 0.18);
        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1E40AF;
        border-color: #1E40AF;
        color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(29, 78, 216, 0.24);
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus,
    div[data-testid="stFormSubmitButton"] > button:focus {
        color: #FFFFFF;
        border-color: #1E40AF;
    }

    /* Professional download buttons */
    div.stDownloadButton > button {
        min-height: 46px;
        border: 1px solid #0F766E !important;
        border-radius: 9px;
        background: linear-gradient(
            135deg,
            #0F766E 0%,
            #115E59 100%
        ) !important;
        color: #FFFFFF !important;
        font-size: 0.91rem;
        font-weight: 750;
        letter-spacing: 0.015em;
        box-shadow:
            0 5px 14px rgba(15, 118, 110, 0.20),
            inset 0 1px 0 rgba(255, 255, 255, 0.16);
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            background 0.18s ease;
    }

    div.stDownloadButton > button p,
    div.stDownloadButton > button span {
        color: #FFFFFF !important;
        font-weight: 750 !important;
    }

    div.stDownloadButton > button:hover {
        border-color: #134E4A !important;
        background: linear-gradient(
            135deg,
            #115E59 0%,
            #134E4A 100%
        ) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
        box-shadow:
            0 8px 20px rgba(15, 118, 110, 0.27),
            inset 0 1px 0 rgba(255, 255, 255, 0.14);
    }

    div.stDownloadButton > button:focus,
    div.stDownloadButton > button:active {
        color: #FFFFFF !important;
        border-color: #134E4A !important;
    }

    /* Primary AI action button */
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 52px;
        border: 0 !important;
        border-radius: 10px;
        background: linear-gradient(
            135deg,
            #F59E0B 0%,
            #EA580C 100%
        ) !important;
        color: #FFFFFF !important;
        font-size: 0.96rem;
        font-weight: 800;
        letter-spacing: 0.025em;
        box-shadow:
            0 7px 18px rgba(234, 88, 12, 0.24),
            inset 0 1px 0 rgba(255, 255, 255, 0.22);
    }

    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(
            135deg,
            #EA580C 0%,
            #C2410C 100%
        ) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
        box-shadow:
            0 10px 24px rgba(194, 65, 12, 0.30),
            inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }

    div[data-testid="stFormSubmitButton"] > button:focus,
    div[data-testid="stFormSubmitButton"] > button:active {
        color: #FFFFFF !important;
        border: 0 !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--slate-200);
        border-radius: 11px;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.035);
    }

    /* Plotly charts */
    div[data-testid="stPlotlyChart"] {
        padding: 0.55rem;
        background: var(--surface);
        border: 1px solid var(--slate-200);
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border-width: 1px;
        box-shadow: none;
    }

    /* Expanders */
    details {
        background: var(--surface);
        border: 1px solid var(--slate-200) !important;
        border-radius: 10px !important;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(
            90deg,
            #2563EB,
            #1D4ED8
        );
    }

    /* Professional hero */
    .lineguard-hero {
        margin-bottom: 1.25rem;
        padding: 1.55rem 1.7rem;
        background:
            linear-gradient(
                125deg,
                #07111F 0%,
                #0F2747 62%,
                #143D68 100%
            );
        border: 1px solid rgba(96, 165, 250, 0.20);
        border-radius: 15px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.14);
    }

    .lineguard-kicker {
        margin-bottom: 0.45rem;
        color: #93C5FD;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .lineguard-hero h1 {
        margin: 0 !important;
        color: #FFFFFF !important;
        font-size: 2.25rem !important;
        letter-spacing: -0.035em;
    }

    .lineguard-hero p {
        max-width: 850px;
        margin-top: 0.65rem;
        margin-bottom: 0;
        color: #CBD5E1;
        font-size: 0.98rem;
    }

    .lineguard-status {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 1rem;
        padding: 0.42rem 0.72rem;
        color: #DBEAFE;
        background: rgba(37, 99, 235, 0.18);
        border: 1px solid rgba(147, 197, 253, 0.24);
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .lineguard-status-dot {
        width: 8px;
        height: 8px;
        background: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.14);
    }

    /* Mobile layout */
    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .lineguard-hero {
            padding: 1.25rem;
        }

        .lineguard-hero h1 {
            font-size: 1.75rem !important;
        }

        div[data-testid="stMetric"] {
            min-height: 105px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Professional Plotly defaults
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#2563EB",
    "#0F766E",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
]



PROJECT_ROOT = Path(__file__).resolve().parent

RISK_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "segment_risk_scores.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_lineguard_model.joblib"
)

LOGISTIC_METRICS_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_metrics.json"
)

XGBOOST_METRICS_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_metrics.json"
)



RISK_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
]

RISK_CHART_ORDER = [
    "Low",
    "Medium",
    "High",
    "Critical",
]

RISK_COLORS = {
    "Critical": "#DC2626",
    "High": "#EA580C",
    "Medium": "#D97706",
    "Low": "#15803D",
}

MODEL_FEATURES = [
    "zone",
    "latitude",
    "longitude",
    "line_age_years",
    "conductor_material",
    "span_length_m",
    "visual_condition_score",
    "vegetation_proximity_score",
    "past_fault_count",
    "months_since_last_inspection",
    "average_load_percentage",
    "overload_hours_last_month",
    "max_wind_speed_kmph",
    "heavy_rainfall_days",
]


@st.cache_data
def load_risk_data() -> pd.DataFrame:
    """Load the final scored LT-line dataset."""

    if not RISK_DATA_PATH.exists():
        raise FileNotFoundError(
            "Risk-score dataset was not found. "
            "Run src/finalize_model.py first."
        )

    return pd.read_csv(RISK_DATA_PATH)


@st.cache_resource
def load_lineguard_model():
    """Load the final trained LineGuard model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Final model was not found. "
            "Run src/finalize_model.py first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics_file(
    file_path: Path,
) -> dict:
    """Load model-evaluation metrics from a JSON file."""

    if not file_path.exists():
        return {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as metrics_file:
        return json.load(metrics_file)


try:
    risk_data = load_risk_data()
    lineguard_model = load_lineguard_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()



def filter_data(
    dataset: pd.DataFrame,
    selected_zones: list[str],
    selected_risks: list[str],
) -> pd.DataFrame:
    """Filter records based on sidebar selections."""

    if not selected_zones or not selected_risks:
        return dataset.iloc[0:0].copy()

    return dataset[
        dataset["zone"].isin(selected_zones)
        & dataset["risk_level"].isin(selected_risks)
    ].copy()


def assign_risk_level(
    risk_score: float,
) -> str:
    """Convert a 0–100 score into a risk category."""

    if risk_score >= 75:
        return "Critical"

    if risk_score >= 55:
        return "High"

    if risk_score >= 30:
        return "Medium"

    return "Low"


def assign_recommendation(
    risk_level: str,
) -> str:
    """Return an inspection recommendation."""

    recommendations = {
        "Critical": "Inspect within 48 hours",
        "High": "Inspect within 7 days",
        "Medium": "Include in the next scheduled inspection",
        "Low": "Continue routine monitoring",
    }

    return recommendations[risk_level]


def clean_feature_name(
    feature_name: str,
) -> str:
    """Convert transformed model features into readable labels."""

    cleaned = (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )

    if cleaned.startswith("zone_"):
        zone_name = cleaned.replace(
            "zone_",
            "",
        )

        return f"Zone: {zone_name}"

    if cleaned.startswith(
        "conductor_material_"
    ):
        material = cleaned.replace(
            "conductor_material_",
            "",
        )

        return f"Conductor Material: {material}"

    return cleaned.replace("_", " ").title()


def explain_segment_prediction(
    segment_row: pd.Series,
) -> pd.DataFrame:
    """
    Calculate feature contributions for the selected
    Logistic Regression model.
    """

    sample = pd.DataFrame(
        [
            {
                feature: segment_row[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )

    preprocessor = lineguard_model.named_steps[
        "preprocessor"
    ]

    classifier = lineguard_model.named_steps[
        "model"
    ]

    transformed_sample = preprocessor.transform(
        sample
    )

    if hasattr(
        transformed_sample,
        "toarray",
    ):
        transformed_sample = (
            transformed_sample.toarray()
        )

    transformed_sample = np.asarray(
        transformed_sample
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    contributions = (
        transformed_sample[0]
        * classifier.coef_[0]
    )

    explanation = pd.DataFrame(
        {
            "Feature": [
                clean_feature_name(name)
                for name in feature_names
            ],
            "Contribution": contributions,
        }
    )

    explanation["Effect"] = np.where(
        explanation["Contribution"] >= 0,
        "Increases Risk",
        "Reduces Risk",
    )

    explanation["Absolute Contribution"] = (
        explanation["Contribution"].abs()
    )

    return explanation.sort_values(
        "Absolute Contribution",
        ascending=False,
    )


def predict_risk_scores(
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Generate model-derived risk scores from 0 to 100."""

    probabilities = (
        lineguard_model.predict_proba(
            input_data[MODEL_FEATURES]
        )[:, 1]
    )

    return probabilities * 100


def prepare_storm_simulation(
    source_data: pd.DataFrame,
    wind_increase: float,
    rainfall_increase: int,
    load_increase: float,
    overload_increase: float,
) -> pd.DataFrame:
    """Apply storm conditions and recalculate line risks."""

    simulation_data = source_data.copy()

    baseline_scores = predict_risk_scores(
        simulation_data
    )

    simulated_features = simulation_data[
        MODEL_FEATURES
    ].copy()

    simulated_features[
        "max_wind_speed_kmph"
    ] = np.clip(
        simulated_features[
            "max_wind_speed_kmph"
        ] + wind_increase,
        5,
        180,
    )

    simulated_features[
        "heavy_rainfall_days"
    ] = np.clip(
        simulated_features[
            "heavy_rainfall_days"
        ] + rainfall_increase,
        0,
        30,
    )

    simulated_features[
        "average_load_percentage"
    ] = np.clip(
        simulated_features[
            "average_load_percentage"
        ] + load_increase,
        20,
        180,
    )

    simulated_features[
        "overload_hours_last_month"
    ] = np.clip(
        simulated_features[
            "overload_hours_last_month"
        ] + overload_increase,
        0,
        250,
    )

    simulated_scores = predict_risk_scores(
        simulated_features
    )

    results = simulation_data[
        [
            "segment_id",
            "zone",
            "latitude",
            "longitude",
            "line_age_years",
            "conductor_material",
            "past_fault_count",
            "visual_condition_score",
            "vegetation_proximity_score",
        ]
    ].copy()

    results["baseline_wind_kmph"] = (
        simulation_data[
            "max_wind_speed_kmph"
        ].values
    )

    results["simulated_wind_kmph"] = (
        simulated_features[
            "max_wind_speed_kmph"
        ].values
    )

    results["baseline_rainfall_days"] = (
        simulation_data[
            "heavy_rainfall_days"
        ].values
    )

    results["simulated_rainfall_days"] = (
        simulated_features[
            "heavy_rainfall_days"
        ].values
    )

    results["baseline_load_percent"] = (
        simulation_data[
            "average_load_percentage"
        ].values
    )

    results["simulated_load_percent"] = (
        simulated_features[
            "average_load_percentage"
        ].values
    )

    results["baseline_overload_hours"] = (
        simulation_data[
            "overload_hours_last_month"
        ].values
    )

    results["simulated_overload_hours"] = (
        simulated_features[
            "overload_hours_last_month"
        ].values
    )

    results["baseline_risk_score"] = (
        baseline_scores.round(1)
    )

    results["simulated_risk_score"] = (
        simulated_scores.round(1)
    )

    results["risk_increase"] = (
        results["simulated_risk_score"]
        - results["baseline_risk_score"]
    ).round(1)

    results["baseline_risk_level"] = (
        results["baseline_risk_score"]
        .apply(assign_risk_level)
    )

    results["simulated_risk_level"] = (
        results["simulated_risk_score"]
        .apply(assign_risk_level)
    )

    results["simulated_recommendation"] = (
        results["simulated_risk_level"]
        .apply(assign_recommendation)
    )

    results["became_high_or_critical"] = (
        results["baseline_risk_level"].isin(
            ["Low", "Medium"]
        )
        &
        results["simulated_risk_level"].isin(
            ["High", "Critical"]
        )
    )

    results["became_critical"] = (
        (
            results["baseline_risk_level"]
            != "Critical"
        )
        &
        (
            results["simulated_risk_level"]
            == "Critical"
        )
    )

    return results.sort_values(
        "simulated_risk_score",
        ascending=False,
    )


def metric_percentage(
    metrics: dict,
    metric_name: str,
) -> float:
    """Read a metric and convert it into percentage form."""

    return float(
        metrics.get(
            metric_name,
            0,
        )
    ) * 100


st.sidebar.title("⚡ LineGuard AI")

st.sidebar.caption(
    "Predictive LT-line maintenance decision support"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "District Overview",
        "Segment Explorer",
        "Predict New Segment",
        "Storm Simulator",
        "Model Performance",
        "About",
    ],
)

st.sidebar.divider()

all_zones = sorted(
    risk_data["zone"].unique()
)

selected_zones = st.sidebar.multiselect(
    "Filter by zone",
    options=all_zones,
    default=all_zones,
)

selected_risks = st.sidebar.multiselect(
    "Filter by risk level",
    options=RISK_ORDER,
    default=RISK_ORDER,
)

filtered_data = filter_data(
    risk_data,
    selected_zones,
    selected_risks,
)

st.sidebar.divider()

st.sidebar.info(
    "Prototype built using a physically informed "
    "synthetic LT-line dataset."
)
if page == "District Overview":

    st.markdown(
        """
<div class="lineguard-hero">
    <div class="lineguard-kicker">
        AI-Powered Infrastructure Intelligence
    </div>
    <h1>LineGuard AI</h1>
    <p>
        Explainable predictive risk scoring for preventive
        low-tension electricity-line maintenance. Prioritise
        inspections using asset condition, electrical stress,
        fault history and environmental exposure.
    </p>
    <div class="lineguard-status">
        <span class="lineguard-status-dot"></span>
        Predictive maintenance model operational
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Risk scores are relative model-derived maintenance "
        "priorities, not guaranteed real-world failure probabilities."
    )

    if filtered_data.empty:
        st.warning(
            "No segments match the selected filters."
        )
        st.stop()

    total_segments = len(filtered_data)

    critical_count = int(
        (
            filtered_data["risk_level"]
            == "Critical"
        ).sum()
    )

    high_count = int(
        (
            filtered_data["risk_level"]
            == "High"
        ).sum()
    )

    average_risk = filtered_data[
        "risk_score"
    ].mean()

    urgent_count = (
        critical_count
        + high_count
    )

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
        metric_5,
    ) = st.columns(5)

    metric_1.metric(
        "Total Segments",
        f"{total_segments:,}",
    )

    metric_2.metric(
        "Critical",
        f"{critical_count:,}",
    )

    metric_3.metric(
        "High Risk",
        f"{high_count:,}",
    )

    metric_4.metric(
        "Average Risk",
        f"{average_risk:.1f}/100",
    )

    metric_5.metric(
        "Priority Inspections",
        f"{urgent_count:,}",
        help=(
            "Critical and High-risk "
            "segments combined"
        ),
    )

    st.divider()

    chart_column, zone_column = (
        st.columns(2)
    )

    with chart_column:

        st.subheader(
            "Risk-Level Distribution"
        )

        risk_distribution = (
            filtered_data["risk_level"]
            .value_counts()
            .reindex(
                RISK_ORDER,
                fill_value=0,
            )
            .reset_index()
        )

        risk_distribution.columns = [
            "Risk Level",
            "Segments",
        ]

        distribution_chart = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Segments",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            category_orders={
                "Risk Level": RISK_ORDER
            },
            text="Segments",
        )

        distribution_chart.update_traces(
            textposition="outside",
        )

        distribution_chart.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Number of Segments",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            distribution_chart,
            width="stretch",
        )

    with zone_column:

        st.subheader(
            "Average Risk by Zone"
        )

        zone_summary = (
            filtered_data
            .groupby(
                "zone",
                as_index=False,
            )
            .agg(
                average_risk=(
                    "risk_score",
                    "mean",
                ),
                total_segments=(
                    "segment_id",
                    "count",
                ),
            )
            .sort_values(
                "average_risk",
                ascending=False,
            )
        )

        zone_chart = px.bar(
            zone_summary,
            x="zone",
            y="average_risk",
            text_auto=".1f",
            hover_data=[
                "total_segments"
            ],
        )

        zone_chart.update_layout(
            xaxis_title="Zone",
            yaxis_title="Average Risk Score",
        )

        st.plotly_chart(
            zone_chart,
            width="stretch",
        )

    st.divider()

    st.subheader(
        "Top Maintenance Priorities"
    )

    number_to_show = st.slider(
        "Number of priority segments to display",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    priority_columns = [
        "priority_rank",
        "segment_id",
        "zone",
        "risk_score",
        "risk_level",
        "line_age_years",
        "past_fault_count",
        "max_wind_speed_kmph",
        "recommendation",
    ]

    priority_table = (
        filtered_data
        .sort_values(
            "risk_score",
            ascending=False,
        )[priority_columns]
        .head(number_to_show)
        .copy()
    )

    priority_table.columns = [
        "Rank",
        "Segment ID",
        "Zone",
        "Risk Score",
        "Risk Level",
        "Line Age",
        "Past Faults",
        "Maximum Wind",
        "Recommended Action",
    ]

    st.dataframe(
        priority_table,
        width="stretch",
        hide_index=True,
    )

    inspection_list = (
        filtered_data[
            filtered_data[
                "risk_level"
            ].isin(
                [
                    "Critical",
                    "High",
                ]
            )
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
    )

    inspection_csv = (
        inspection_list
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇ Download Priority Inspection List",
        data=inspection_csv,
        file_name=(
            "lineguard_priority_inspections.csv"
        ),
        mime="text/csv",
    )




elif page == "Segment Explorer":

    st.title("Segment Explorer")

    st.write(
        "Select an LT-line segment to inspect its "
        "condition, risk score, maintenance priority "
        "and AI explanation."
    )

    if filtered_data.empty:
        st.warning(
            "No segments match the selected filters. "
            "Change the sidebar filters."
        )
        st.stop()

    explorer_data = (
        filtered_data
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .copy()
    )

    segment_options = (
        explorer_data["segment_id"]
        .astype(str)
        .tolist()
    )

    segment_labels = {}

    for _, row in explorer_data.iterrows():
        segment_labels[
            str(row["segment_id"])
        ] = (
            f"{row['segment_id']} — "
            f"{row['risk_score']:.1f}/100 — "
            f"{row['risk_level']}"
        )

    selected_segment_id = st.selectbox(
        "Select a line segment",
        options=segment_options,
        format_func=lambda segment_id: (
            segment_labels[segment_id]
        ),
    )

    selected_segment = (
        explorer_data[
            explorer_data["segment_id"]
            == selected_segment_id
        ]
        .iloc[0]
    )

    risk_score = float(
        selected_segment["risk_score"]
    )

    risk_level = str(
        selected_segment["risk_level"]
    )

    recommendation = str(
        selected_segment["recommendation"]
    )

    district_rank = int(
        selected_segment["priority_rank"]
    )

    st.divider()

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
    ) = st.columns(4)

    metric_1.metric(
        "Segment ID",
        selected_segment_id,
    )

    metric_2.metric(
        "Risk Score",
        f"{risk_score:.1f}/100",
    )

    metric_3.metric(
        "Risk Level",
        risk_level,
    )

    metric_4.metric(
        "District Rank",
        f"#{district_rank}",
    )

    st.progress(
        min(
            max(
                risk_score / 100,
                0.0,
            ),
            1.0,
        )
    )

    if risk_level == "Critical":
        st.error(
            "Critical maintenance priority: "
            f"{recommendation}"
        )

    elif risk_level == "High":
        st.warning(
            "High maintenance priority: "
            f"{recommendation}"
        )

    elif risk_level == "Medium":
        st.info(
            "Medium maintenance priority: "
            f"{recommendation}"
        )

    else:
        st.success(
            "Low maintenance priority: "
            f"{recommendation}"
        )

    st.divider()

    asset_column, stress_column = (
        st.columns(2)
    )

    with asset_column:

        st.subheader(
            "Asset and Inspection Details"
        )

        asset_details = pd.DataFrame(
            {
                "Attribute": [
                    "Zone",
                    "Line age",
                    "Conductor material",
                    "Span length",
                    "Visual condition",
                    "Vegetation proximity",
                    "Past fault count",
                    "Months since inspection",
                ],
                "Value": [
                    selected_segment["zone"],
                    (
                        f"{int(selected_segment['line_age_years'])} "
                        "years"
                    ),
                    selected_segment[
                        "conductor_material"
                    ],
                    (
                        f"{selected_segment['span_length_m']:.1f} m"
                    ),
                    (
                        f"{int(selected_segment['visual_condition_score'])}/10"
                    ),
                    (
                        f"{int(selected_segment['vegetation_proximity_score'])}/10"
                    ),
                    int(
                        selected_segment[
                            "past_fault_count"
                        ]
                    ),
                    int(
                        selected_segment[
                            "months_since_last_inspection"
                        ]
                    ),
                ],
            }
        )

        st.dataframe(
            asset_details,
            width="stretch",
            hide_index=True,
        )

    with stress_column:

        st.subheader(
            "Electrical and Environmental Stress"
        )

        stress_details = pd.DataFrame(
            {
                "Attribute": [
                    "Average load",
                    "Overload hours last month",
                    "Maximum recent wind",
                    "Heavy rainfall days",
                    "Latitude",
                    "Longitude",
                ],
                "Value": [
                    (
                        f"{selected_segment['average_load_percentage']:.1f}%"
                    ),
                    (
                        f"{selected_segment['overload_hours_last_month']:.1f} hours"
                    ),
                    (
                        f"{selected_segment['max_wind_speed_kmph']:.1f} km/h"
                    ),
                    int(
                        selected_segment[
                            "heavy_rainfall_days"
                        ]
                    ),
                    selected_segment[
                        "latitude"
                    ],
                    selected_segment[
                        "longitude"
                    ],
                ],
            }
        )

        st.dataframe(
            stress_details,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "AI Risk Explanation"
    )

    st.caption(
        "Positive values increase the model risk score. "
        "Negative values reduce it."
    )

    explanation = explain_segment_prediction(
        selected_segment
    )

    positive_factors = (
        explanation[
            explanation["Contribution"] > 0
        ]
        .head(7)
        .sort_values(
            "Contribution",
            ascending=True,
        )
    )

    protective_factors = (
        explanation[
            explanation["Contribution"] < 0
        ]
        .head(5)
        .sort_values(
            "Contribution",
            ascending=False,
        )
    )

    (
        explanation_column,
        protection_column,
    ) = st.columns(2)

    with explanation_column:

        st.markdown(
            "#### Main Risk Drivers"
        )

        if positive_factors.empty:
            st.info(
                "No strong positive risk "
                "contributors were identified."
            )

        else:
            positive_chart = px.bar(
                positive_factors,
                x="Contribution",
                y="Feature",
                orientation="h",
                text="Contribution",
            )

            positive_chart.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
            )

            st.plotly_chart(
                positive_chart,
                width="stretch",
            )

    with protection_column:

        st.markdown(
            "#### Risk-Reducing Factors"
        )

        if protective_factors.empty:
            st.info(
                "No significant risk-reducing "
                "factors were identified."
            )

        else:
            protective_chart = px.bar(
                protective_factors,
                x="Contribution",
                y="Feature",
                orientation="h",
                text="Contribution",
            )

            protective_chart.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
            )

            st.plotly_chart(
                protective_chart,
                width="stretch",
            )

    st.divider()

    map_column, action_column = (
        st.columns(
            [1.2, 1]
        )
    )

    with map_column:

        st.subheader(
            "Segment Location"
        )

        location_data = pd.DataFrame(
            {
                "lat": [
                    selected_segment[
                        "latitude"
                    ]
                ],
                "lon": [
                    selected_segment[
                        "longitude"
                    ]
                ],
            }
        )

        st.map(
            location_data,
            zoom=11,
        )

    with action_column:

        st.subheader(
            "Maintenance Decision"
        )

        st.markdown(
            f"""
**Segment:** {selected_segment_id}

**Relative risk:** {risk_score:.1f}/100

**Risk category:** {risk_level}

**Recommended action:** {recommendation}
"""
        )

        st.warning(
            "The recommendation must be reviewed "
            "by qualified utility personnel."
        )


elif page == "Predict New Segment":

    st.title("Predict New Segment")

    st.write(
        "Enter new LT-line conditions and generate "
        "a live AI maintenance-risk assessment."
    )

    st.warning(
        "The result is a relative maintenance-priority "
        "score, not a certified engineering assessment."
    )

    with st.form(
        "new_segment_form"
    ):

        st.subheader(
            "Location and Asset Details"
        )

        (
            location_col_1,
            location_col_2,
            location_col_3,
        ) = st.columns(3)

        with location_col_1:

            new_zone = st.selectbox(
                "Zone",
                options=[
                    "North",
                    "South",
                    "East",
                    "West",
                    "Central",
                ],
            )

        with location_col_2:

            new_latitude = st.number_input(
                "Latitude",
                min_value=8.0,
                max_value=38.0,
                value=13.0000,
                step=0.0001,
                format="%.4f",
            )

        with location_col_3:

            new_longitude = st.number_input(
                "Longitude",
                min_value=68.0,
                max_value=98.0,
                value=80.1000,
                step=0.0001,
                format="%.4f",
            )

        (
            asset_col_1,
            asset_col_2,
            asset_col_3,
        ) = st.columns(3)

        with asset_col_1:

            new_line_age = st.slider(
                "Line age in years",
                min_value=1,
                max_value=50,
                value=15,
            )

            new_visual_condition = st.slider(
                "Visual condition score",
                min_value=1,
                max_value=10,
                value=5,
            )

        with asset_col_2:

            new_material = st.selectbox(
                "Conductor material",
                options=[
                    "AAC",
                    "ACSR",
                    "Copper",
                ],
            )

            new_vegetation = st.slider(
                "Vegetation proximity score",
                min_value=0,
                max_value=10,
                value=4,
            )

        with asset_col_3:

            new_span_length = st.number_input(
                "Span length in metres",
                min_value=10.0,
                max_value=100.0,
                value=35.0,
                step=1.0,
            )

            new_past_faults = st.number_input(
                "Previous fault count",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
            )

        st.divider()

        (
            stress_col_1,
            stress_col_2,
            stress_col_3,
        ) = st.columns(3)

        with stress_col_1:

            new_months_since_inspection = st.slider(
                "Months since last inspection",
                min_value=1,
                max_value=60,
                value=12,
            )

        with stress_col_2:

            new_average_load = st.slider(
                "Average load percentage",
                min_value=30,
                max_value=150,
                value=75,
            )

        with stress_col_3:

            new_overload_hours = st.number_input(
                "Overload hours last month",
                min_value=0.0,
                max_value=150.0,
                value=10.0,
                step=1.0,
            )

        st.divider()

        weather_col_1, weather_col_2 = (
            st.columns(2)
        )

        with weather_col_1:

            new_max_wind = st.slider(
                "Maximum recent wind speed in km/h",
                min_value=5,
                max_value=150,
                value=40,
            )

        with weather_col_2:

            new_rainfall_days = st.slider(
                "Heavy rainfall days",
                min_value=0,
                max_value=20,
                value=3,
            )

        submitted = st.form_submit_button(
            "⚡ Generate AI Risk Assessment",
            width="stretch",
        )

    if submitted:

        new_segment_data = pd.DataFrame(
            [
                {
                    "zone": new_zone,
                    "latitude": float(
                        new_latitude
                    ),
                    "longitude": float(
                        new_longitude
                    ),
                    "line_age_years": int(
                        new_line_age
                    ),
                    "conductor_material": (
                        new_material
                    ),
                    "span_length_m": float(
                        new_span_length
                    ),
                    "visual_condition_score": int(
                        new_visual_condition
                    ),
                    "vegetation_proximity_score": int(
                        new_vegetation
                    ),
                    "past_fault_count": int(
                        new_past_faults
                    ),
                    "months_since_last_inspection": int(
                        new_months_since_inspection
                    ),
                    "average_load_percentage": float(
                        new_average_load
                    ),
                    "overload_hours_last_month": float(
                        new_overload_hours
                    ),
                    "max_wind_speed_kmph": float(
                        new_max_wind
                    ),
                    "heavy_rainfall_days": int(
                        new_rainfall_days
                    ),
                }
            ]
        )

        model_risk_value = (
            lineguard_model.predict_proba(
                new_segment_data[
                    MODEL_FEATURES
                ]
            )[0, 1]
        )

        new_risk_score = round(
            float(model_risk_value) * 100,
            1,
        )

        new_risk_level = assign_risk_level(
            new_risk_score
        )

        new_recommendation = assign_recommendation(
            new_risk_level
        )

        st.divider()

        st.subheader(
            "AI Assessment Result"
        )

        result_col_1, result_col_2 = (
            st.columns(2)
        )

        result_col_1.metric(
            "Relative Risk Score",
            f"{new_risk_score:.1f}/100",
        )

        result_col_2.metric(
            "Risk Level",
            new_risk_level,
        )

        st.markdown(
            f"### Recommended Action: {new_recommendation}"
        )

        st.progress(
            new_risk_score / 100
        )

        explanation = explain_segment_prediction(
            new_segment_data.iloc[0]
        )

        main_risk_factors = (
            explanation[
                explanation[
                    "Contribution"
                ] > 0
            ]
            .head(7)
            .sort_values(
                "Contribution",
                ascending=True,
            )
        )

        st.subheader(
            "Why Did the AI Assign This Score?"
        )

        if main_risk_factors.empty:
            st.info(
                "No strong risk-increasing "
                "factors were detected."
            )

        else:
            prediction_chart = px.bar(
                main_risk_factors,
                x="Contribution",
                y="Feature",
                orientation="h",
                text="Contribution",
            )

            prediction_chart.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
            )

            st.plotly_chart(
                prediction_chart,
                width="stretch",
            )

elif page == "Storm Simulator":

    st.title("Storm Impact Simulator")

    st.write(
        "Simulate severe weather and increased "
        "electrical demand to identify vulnerable segments."
    )

    if filtered_data.empty:
        st.warning(
            "No segments match the selected filters."
        )
        st.stop()

    with st.form(
        "storm_simulation_form"
    ):

        storm_column_1, storm_column_2 = (
            st.columns(2)
        )

        with storm_column_1:

            wind_increase = st.slider(
                "Additional maximum wind speed",
                0,
                100,
                35,
                5,
            )

            rainfall_increase = st.slider(
                "Additional heavy-rainfall days",
                0,
                15,
                4,
            )

        with storm_column_2:

            load_increase = st.slider(
                "Electrical-load increase",
                0,
                60,
                20,
                5,
            )

            overload_increase = st.slider(
                "Additional overload hours",
                0,
                120,
                30,
                5,
            )

        run_simulation = st.form_submit_button(
            "Run Storm Risk Simulation",
            width="stretch",
        )

    if run_simulation:

        simulation_results = (
            prepare_storm_simulation(
                source_data=filtered_data,
                wind_increase=float(
                    wind_increase
                ),
                rainfall_increase=int(
                    rainfall_increase
                ),
                load_increase=float(
                    load_increase
                ),
                overload_increase=float(
                    overload_increase
                ),
            )
        )

        baseline_critical = int(
            (
                simulation_results[
                    "baseline_risk_level"
                ]
                == "Critical"
            ).sum()
        )

        simulated_critical = int(
            (
                simulation_results[
                    "simulated_risk_level"
                ]
                == "Critical"
            ).sum()
        )

        newly_critical = int(
            simulation_results[
                "became_critical"
            ].sum()
        )

        newly_priority = int(
            simulation_results[
                "became_high_or_critical"
            ].sum()
        )

        baseline_average = (
            simulation_results[
                "baseline_risk_score"
            ].mean()
        )

        simulated_average = (
            simulation_results[
                "simulated_risk_score"
            ].mean()
        )

        (
            result_1,
            result_2,
            result_3,
            result_4,
        ) = st.columns(4)

        result_1.metric(
            "Critical Before",
            baseline_critical,
        )

        result_2.metric(
            "Critical After",
            simulated_critical,
            delta=(
                simulated_critical
                - baseline_critical
            ),
            delta_color="inverse",
        )

        result_3.metric(
            "Newly Critical",
            newly_critical,
        )

        result_4.metric(
            "New Priority Segments",
            newly_priority,
        )

        st.metric(
            "Average Risk After Storm",
            f"{simulated_average:.1f}/100",
            delta=(
                f"{simulated_average - baseline_average:+.1f}"
            ),
            delta_color="inverse",
        )

        baseline_distribution = (
            simulation_results[
                "baseline_risk_level"
            ]
            .value_counts()
            .reindex(
                RISK_CHART_ORDER,
                fill_value=0,
            )
        )

        simulated_distribution = (
            simulation_results[
                "simulated_risk_level"
            ]
            .value_counts()
            .reindex(
                RISK_CHART_ORDER,
                fill_value=0,
            )
        )

        comparison_data = (
            pd.DataFrame(
                {
                    "Risk Level": (
                        RISK_CHART_ORDER
                    ),
                    "Before Storm": (
                        baseline_distribution.values
                    ),
                    "After Storm": (
                        simulated_distribution.values
                    ),
                }
            )
            .melt(
                id_vars="Risk Level",
                var_name="Scenario",
                value_name="Segments",
            )
        )

        comparison_chart = px.bar(
            comparison_data,
            x="Risk Level",
            y="Segments",
            color="Scenario",
            barmode="group",
            text="Segments",
        )

        st.plotly_chart(
            comparison_chart,
            width="stretch",
        )

        st.subheader(
            "Highest-Priority Segments After Storm"
        )

        storm_table = simulation_results[
            [
                "segment_id",
                "zone",
                "baseline_risk_score",
                "simulated_risk_score",
                "risk_increase",
                "baseline_risk_level",
                "simulated_risk_level",
                "simulated_recommendation",
            ]
        ].head(25).copy()

        storm_table.columns = [
            "Segment ID",
            "Zone",
            "Risk Before",
            "Risk After",
            "Increase",
            "Level Before",
            "Level After",
            "Recommended Action",
        ]

        st.dataframe(
            storm_table,
            width="stretch",
            hide_index=True,
        )

        simulation_csv = (
            simulation_results
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇ Download Storm Simulation Report",
            data=simulation_csv,
            file_name=(
                "lineguard_storm_simulation.csv"
            ),
            mime="text/csv",
        )


elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Two machine-learning models were evaluated. "
        "The final model was chosen using safety-focused "
        "metrics rather than accuracy alone."
    )

    logistic_metrics = load_metrics_file(
        LOGISTIC_METRICS_PATH
    )

    xgboost_metrics = load_metrics_file(
        XGBOOST_METRICS_PATH
    )

    if not logistic_metrics:
        st.error(
            "Logistic Regression metrics are missing. "
            "Run src/train_baseline.py."
        )
        st.stop()

    if not xgboost_metrics:
        st.error(
            "XGBoost metrics are missing. "
            "Run src/train_xgboost.py."
        )
        st.stop()

    st.success(
        "Selected final model: Logistic Regression. "
        "It detected more upcoming failure cases and "
        "produced stronger overall safety-focused metrics."
    )

    selected_1, selected_2, selected_3, selected_4 = (
        st.columns(4)
    )

    selected_1.metric(
        "Selected Model",
        "Logistic Regression",
    )

    selected_2.metric(
        "Failure Recall",
        (
            f"{metric_percentage(
                logistic_metrics,
                'recall'
            ):.1f}%"
        ),
    )

    selected_3.metric(
        "PR-AUC",
        (
            f"{metric_percentage(
                logistic_metrics,
                'pr_auc'
            ):.1f}%"
        ),
    )

    selected_4.metric(
        "Top 10% Capture",
        (
            f"{metric_percentage(
                logistic_metrics,
                'top_10_percent_risk_capture'
            ):.1f}%"
        ),
    )

    st.divider()

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
                "PR-AUC",
                "Top 10% Capture",
            ],
            "Logistic Regression": [
                metric_percentage(
                    logistic_metrics,
                    "accuracy",
                ),
                metric_percentage(
                    logistic_metrics,
                    "precision",
                ),
                metric_percentage(
                    logistic_metrics,
                    "recall",
                ),
                metric_percentage(
                    logistic_metrics,
                    "f1_score",
                ),
                metric_percentage(
                    logistic_metrics,
                    "roc_auc",
                ),
                metric_percentage(
                    logistic_metrics,
                    "pr_auc",
                ),
                metric_percentage(
                    logistic_metrics,
                    "top_10_percent_risk_capture",
                ),
            ],
            "XGBoost": [
                metric_percentage(
                    xgboost_metrics,
                    "accuracy",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "precision",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "recall",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "f1_score",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "roc_auc",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "pr_auc",
                ),
                metric_percentage(
                    xgboost_metrics,
                    "top_10_percent_risk_capture",
                ),
            ],
        }
    )

    formatted_comparison = comparison.copy()

    formatted_comparison[
        "Logistic Regression"
    ] = formatted_comparison[
        "Logistic Regression"
    ].map(
        lambda value: f"{value:.1f}%"
    )

    formatted_comparison[
        "XGBoost"
    ] = formatted_comparison[
        "XGBoost"
    ].map(
        lambda value: f"{value:.1f}%"
    )

    st.subheader(
        "Model Comparison Table"
    )

    st.dataframe(
        formatted_comparison,
        width="stretch",
        hide_index=True,
    )

    chart_data = comparison.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score",
    )

    comparison_chart = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
    )

    comparison_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    comparison_chart.update_layout(
        yaxis_title="Score (%)",
        yaxis_range=[
            0,
            100,
        ],
    )

    st.plotly_chart(
        comparison_chart,
        width="stretch",
    )

    st.divider()

    st.subheader(
        "Confusion Matrices"
    )

    logistic_matrix = np.array(
        logistic_metrics[
            "confusion_matrix"
        ]
    )

    xgboost_matrix = np.array(
        xgboost_metrics[
            "confusion_matrix"
        ]
    )

    matrix_column_1, matrix_column_2 = (
        st.columns(2)
    )

    with matrix_column_1:

        st.markdown(
            "#### Logistic Regression"
        )

        logistic_matrix_chart = px.imshow(
            logistic_matrix,
            text_auto=True,
            x=[
                "Predicted Safe",
                "Predicted Failure",
            ],
            y=[
                "Actual Safe",
                "Actual Failure",
            ],
            labels={
                "x": "Prediction",
                "y": "Actual Class",
                "color": "Segments",
            },
        )

        st.plotly_chart(
            logistic_matrix_chart,
            width="stretch",
        )

        st.caption(
            "The model correctly identified "
            f"{int(logistic_matrix[1, 1])} of "
            f"{int(logistic_matrix[1].sum())} "
            "upcoming failure cases."
        )

    with matrix_column_2:

        st.markdown(
            "#### XGBoost"
        )

        xgboost_matrix_chart = px.imshow(
            xgboost_matrix,
            text_auto=True,
            x=[
                "Predicted Safe",
                "Predicted Failure",
            ],
            y=[
                "Actual Safe",
                "Actual Failure",
            ],
            labels={
                "x": "Prediction",
                "y": "Actual Class",
                "color": "Segments",
            },
        )

        st.plotly_chart(
            xgboost_matrix_chart,
            width="stretch",
        )

        st.caption(
            "The model correctly identified "
            f"{int(xgboost_matrix[1, 1])} of "
            f"{int(xgboost_matrix[1].sum())} "
            "upcoming failure cases."
        )

    st.divider()

    st.subheader(
        "Why Logistic Regression Was Selected"
    )

    st.markdown(
        """
For preventive electricity-line maintenance, missing a
genuinely vulnerable line can be more serious than scheduling
an unnecessary inspection.

Although XGBoost achieved higher overall accuracy and precision,
Logistic Regression achieved:

- Higher recall for failure cases
- Higher F1-score
- Higher ROC-AUC
- Higher PR-AUC
- Better capture of failures among the highest-risk segments

The simpler model was therefore selected because it better
supports the project's preventive-safety objective.
"""
    )

    with st.expander(
        "Understand the evaluation metrics"
    ):

        st.markdown(
            """
**Accuracy:** Percentage of all segments classified correctly.

**Precision:** Among segments flagged as failures, how many
actually belonged to the failure class.

**Recall:** Among actual failure cases, how many the model detected.

**F1-score:** Balance between precision and recall.

**ROC-AUC:** Ability to rank failure cases above safer segments.

**PR-AUC:** Precision-recall performance for the minority
failure class.

**Top 10% capture:** Percentage of all failure cases contained
within the model's highest-risk 10% of segments.
"""
        )

    st.warning(
        "These results are based on a synthetic hackathon "
        "dataset. Real-world validation requires utility asset, "
        "inspection, outage and verified failure records."
    )


elif page == "About":

    st.title("About LineGuard AI")

    st.write(
        "LineGuard AI is an explainable "
        "predictive-maintenance prototype for "
        "low-tension electricity distribution lines."
    )

    st.markdown(
        """
### Current capabilities

- District-level risk overview
- Priority inspection ranking
- Individual segment exploration
- Explainable AI risk analysis
- Live prediction for new segments
- Storm and electrical-load simulation
- Logistic Regression and XGBoost comparison
- Downloadable maintenance reports

### Intended impact

The system helps maintenance teams identify which line
segments should be inspected first, supporting a shift from
reactive emergency repair to preventive maintenance.
"""
    )

    st.warning(
        "LineGuard AI does not replace circuit breakers, "
        "protection systems, utility engineers, physical "
        "inspection or established safety procedures."
    )

    st.info(
        "The current prototype uses a physically informed "
        "synthetic dataset because actual electricity-board "
        "asset and failure data were unavailable."
    )
