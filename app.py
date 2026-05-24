from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xgboost_churn_model.json"
PREPROCESSOR_PATH = APP_DIR / "preprocessor.pkl"
SCHEMA_PATH = APP_DIR / "app_schema.json"

BUSINESS_THRESHOLD = 0.099
FN_COST = 40_000
FP_COST = 500

BG = "#F8F1E7"
PANEL = "#FFF9F0"
PANEL_SOFT = "#FCF4EA"
INK = "#2B2118"
MUTED = "#705E50"
RUST = "#A84F2A"
AMBER = "#C9893F"
SAGE = "#55715B"
CLAY = "#D9B99A"
TERRACOTTA = "#C87554"
RULE = "#E5D4C0"


st.set_page_config(page_title="CRIS Dashboard", layout="wide")


def apply_style() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="stApp"] {{
            background: {BG};
            color: {INK};
            font-family: Arial, Helvetica, sans-serif;
        }}
        .block-container {{
            max-width: 1440px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }}
        p, label, span, div {{
            color: {INK};
        }}
        h1, h2, h3 {{
            color: {INK};
            letter-spacing: 0;
        }}
        h1 {{
            font-size: 2.15rem;
            line-height: 1.08;
            margin-bottom: 0.25rem;
        }}
        h2 {{
            font-size: 1.35rem;
        }}
        h3 {{
            color: {RUST};
            font-size: 1rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        #MainMenu,
        .stDeployButton {{
            visibility: hidden;
            display: none;
            height: 0;
        }}
        div[data-testid="stSidebar"] {{
            background: #EFE1D0;
            border-right: 1px solid {RULE};
        }}
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="collapsedControl"],
        button[data-testid="baseButton-header"],
        button[kind="header"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarNav"] button {{
            background: {PANEL} !important;
            border: 1px solid {CLAY} !important;
            color: {INK} !important;
        }}
        button[data-testid="stSidebarCollapseButton"] svg,
        button[data-testid="collapsedControl"] svg,
        button[data-testid="baseButton-header"] svg,
        button[kind="header"] svg,
        [data-testid="stSidebarCollapsedControl"] svg {{
            color: {INK} !important;
            fill: {INK} !important;
        }}
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] label {{
            color: {MUTED};
        }}
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {{
            background: #EDE2D0 !important;
            color: {INK} !important;
            border: 1px solid #E0D2BF !important;
            border-radius: 7px !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background: #EDE2D0 !important;
            color: {INK} !important;
            border: 0 !important;
        }}
        div[data-testid="stNumberInput"] button svg {{
            color: {INK} !important;
            fill: {INK} !important;
        }}
        div[data-testid="stSlider"] div[role="slider"] {{
            background: {RUST} !important;
            border-color: {RUST} !important;
        }}
        div[data-testid="stSlider"] [data-testid="stTickBar"] {{
            background: #E7D7C5 !important;
        }}
        div[data-testid="stSlider"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {{
            color: {INK} !important;
            font-weight: 600;
        }}
        div[data-baseweb="select"] > div {{
            background: {PANEL} !important;
            border: 1px solid {CLAY} !important;
            color: {INK} !important;
            border-radius: 7px !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {{
            color: {INK} !important;
            fill: {INK} !important;
        }}
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"] {{
            background: {PANEL} !important;
            border: 1px solid {CLAY} !important;
            color: {INK} !important;
            box-shadow: 0 12px 32px rgba(43, 33, 24, 0.16) !important;
        }}
        li[role="option"],
        div[role="option"] {{
            background: {PANEL} !important;
            color: {INK} !important;
        }}
        li[role="option"]:hover,
        div[role="option"]:hover {{
            background: #EFE1D0 !important;
            color: {INK} !important;
        }}
        div[data-testid="stFileUploader"] section {{
            background: {PANEL};
            border: 1px dashed {CLAY};
            border-radius: 8px;
        }}
        div[data-testid="stFileUploader"] section p,
        div[data-testid="stFileUploader"] section span,
        div[data-testid="stFileUploader"] section small {{
            color: {INK} !important;
        }}
        div[data-testid="stFileUploader"] section button {{
            background: {RUST} !important;
            color: #FFF9F0 !important;
            border: 1px solid {RUST} !important;
            border-radius: 7px !important;
        }}
        div[data-testid="stFileUploader"] section button span {{
            color: #FFF9F0 !important;
        }}
        div[data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid {RULE};
            border-top: 3px solid {RUST};
            padding: 0.85rem 0.9rem;
            border-radius: 8px;
            min-height: 7rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {MUTED};
            font-size: 0.78rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {INK};
            font-size: 1.45rem;
        }}
        .hero {{
            background: {PANEL};
            border: 1px solid {RULE};
            border-top: 5px solid {RUST};
            border-radius: 8px;
            padding: 1.35rem 1.45rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .hero-kicker {{
            color: {RUST};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .hero-subtitle {{
            color: {MUTED};
            font-size: 1.02rem;
            max-width: 980px;
            line-height: 1.55;
        }}
        .note {{
            color: {MUTED};
            font-size: 0.9rem;
            line-height: 1.45;
        }}
        .chart-note {{
            color: {MUTED};
            font-size: 0.82rem;
            line-height: 1.42;
            min-height: 2.75rem;
            border-top: 1px solid {RULE};
            margin-top: 0.25rem;
            padding-top: 0.55rem;
        }}
        .chart-shell {{
            background: {PANEL};
            border: 1px solid {RULE};
            border-radius: 8px;
            padding: 0.65rem 0.75rem 0.75rem;
            min-height: 28.5rem;
        }}
        .chart-shell-tall {{
            min-height: 30.5rem;
        }}
        .callout {{
            background: {PANEL};
            border: 1px solid {RULE};
            border-left: 4px solid {AMBER};
            border-radius: 8px;
            padding: 0.95rem 1.05rem;
            line-height: 1.5;
        }}
        .ok {{
            background: #F4F7EF;
            border: 1px solid #CFDBC8;
            color: #334D35;
            padding: 0.62rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.45rem;
            font-size: 0.9rem;
        }}
        .warn {{
            background: #FFF5E5;
            border: 1px solid #E9C99E;
            color: #6F421B;
            padding: 0.62rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.45rem;
            font-size: 0.9rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            background: {RUST};
            color: white;
            border: 1px solid {RUST};
            border-radius: 7px;
            font-weight: 700;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: #8C3E22;
            color: white;
            border-color: #8C3E22;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid {RULE};
        }}
        .stTabs [data-baseweb="tab"] {{
            background: #EFE1D0;
            border: 1px solid {RULE};
            border-bottom: 0;
            border-radius: 7px 7px 0 0;
            color: {MUTED};
            padding: 0.55rem 0.75rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PANEL};
            border-top: 3px solid {RUST};
            color: {INK};
            font-weight: 700;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {RULE};
            border-radius: 8px;
            overflow: hidden;
            background: {PANEL};
        }}
        .warm-table {{
            width: 100%;
            border-collapse: collapse;
            background: {PANEL};
            border: 1px solid {RULE};
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.92rem;
            margin: 0.4rem 0 1rem;
        }}
        .warm-table th {{
            background: #EFE1D0;
            color: {INK};
            text-align: left;
            padding: 0.7rem 0.75rem;
            border-bottom: 1px solid {RULE};
            font-weight: 700;
        }}
        .warm-table td {{
            color: {INK};
            padding: 0.62rem 0.75rem;
            border-bottom: 1px solid #EADCCB;
        }}
        .warm-table tr:nth-child(even) td {{
            background: {PANEL_SOFT};
        }}
        .warm-table-wrap {{
            max-height: 28rem;
            overflow: auto;
            border-radius: 8px;
        }}
        div[data-testid="stCaptionContainer"] p {{
            color: {MUTED};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_assets() -> tuple[Any, xgb.XGBClassifier, dict[str, Any]]:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    return preprocessor, model, schema


def expected_model_columns(preprocessor: Any, schema: dict[str, Any]) -> list[str]:
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return list(schema.keys())


def schema_default_row(schema: dict[str, Any]) -> dict[str, Any]:
    return {feature: details.get("default") for feature, details in schema.items()}


def schema_numeric_bounds(schema: dict[str, Any], feature: str) -> tuple[float, float, float]:
    details = schema.get(feature, {})
    default = float(details.get("default", 0) or 0)
    min_value = float(details.get("min", min(0, default)) or 0)
    max_value = float(details.get("max", max(default * 2, default + 1)) or 1)
    if max_value <= min_value:
        max_value = min_value + 1
    return min_value, max_value, default


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    eps = 1e-5

    if {"monthly_transaction_count", "tenure_months"}.issubset(data.columns):
        data["transaction_frequency"] = data["monthly_transaction_count"] / (data["tenure_months"] + eps)

    if {"avg_monthly_balance", "annual_income"}.issubset(data.columns):
        data["balance_to_income_ratio"] = data["avg_monthly_balance"] / (data["annual_income"] + eps)

    if {"total_complaints", "complaint_resolution_time"}.issubset(data.columns):
        data["complaint_severity_index"] = data["total_complaints"] * data["complaint_resolution_time"]

    if {"mobile_app_login_count", "digital_transaction_ratio"}.issubset(data.columns):
        data["digital_engagement_score"] = data["mobile_app_login_count"] * data["digital_transaction_ratio"]

    return data


def has_predictions(df: pd.DataFrame) -> bool:
    return {"churn_prediction", "churn_probability"}.issubset(df.columns)


def validate_for_prediction(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    source_cols = set(engineer_features(df).columns)
    ignore = {"churn", "churn_prediction", "churn_probability"}
    return [col for col in required_cols if col not in source_cols and col not in ignore]


def clean_feature_name(feature_name: str, source_columns: list[str]) -> str:
    cleaned = feature_name
    for prefix in ("num__", "nom__", "ord__", "cat__"):
        cleaned = cleaned.replace(prefix, "")

    for col in sorted(source_columns, key=len, reverse=True):
        if cleaned == col or cleaned.startswith(f"{col}_"):
            return col
    return cleaned


def friendly_feature_name(feature_name: str) -> str:
    if not feature_name:
        return "Mixed risk signals"
    return feature_name.replace("_", " ").title()


def strongest_business_signal(row: pd.Series) -> str:
    signals = [
        ("unresolved_complaint_count", float(row.get("unresolved_complaint_count", 0) or 0), 1),
        ("complaint_resolution_time", float(row.get("complaint_resolution_time", 0) or 0), 7),
        ("balance_decline_percentage", float(row.get("balance_decline_percentage", 0) or 0), 10),
        ("last_login_days", float(row.get("last_login_days", 0) or 0), 21),
        ("last_contacted_days", float(row.get("last_contacted_days", 0) or 0), 60),
        ("campaign_response_count", 1 - float(row.get("campaign_response_count", 1) or 0), 1),
    ]
    scored_signals = [
        (name, value / threshold)
        for name, value, threshold in signals
        if threshold and value >= threshold
    ]
    if not scored_signals:
        return "Mixed risk signals"
    return friendly_feature_name(max(scored_signals, key=lambda item: item[1])[0])


def xgboost_top_features(
    X: pd.DataFrame,
    processed: Any,
    preprocessor: Any,
    model: xgb.XGBClassifier,
    required_cols: list[str],
) -> list[str]:
    try:
        feature_names = list(preprocessor.get_feature_names_out())
        booster = model.get_booster()
        contributions = booster.predict(xgb.DMatrix(processed), pred_contribs=True)
        contributions = np.asarray(contributions)
        if contributions.ndim != 2 or contributions.shape[1] < 2:
            raise ValueError("Unexpected contribution matrix shape.")
        feature_contribs = contributions[:, :-1]
        top_indexes = np.argmax(feature_contribs, axis=1)
        labels = []
        for idx in top_indexes:
            processed_name = feature_names[int(idx)] if int(idx) < len(feature_names) else ""
            labels.append(friendly_feature_name(clean_feature_name(processed_name, required_cols)))
        return labels
    except Exception:
        return [strongest_business_signal(row) for _, row in X.iterrows()]


def add_top_feature_column(
    scored: pd.DataFrame,
    raw_df: pd.DataFrame,
    preprocessor: Any,
    model: xgb.XGBClassifier,
    required_cols: list[str],
) -> pd.DataFrame:
    output = scored.copy()
    missing = validate_for_prediction(raw_df, required_cols)
    if missing:
        if "top_churn_feature" not in output.columns:
            output["top_churn_feature"] = output.apply(strongest_business_signal, axis=1)
        return output

    engineered = engineer_features(raw_df)
    X = engineered.reindex(columns=required_cols)
    processed = preprocessor.transform(X)
    output["top_churn_feature"] = xgboost_top_features(X, processed, preprocessor, model, required_cols)
    return output


def score_single_customer(
    input_dict: dict[str, Any],
    preprocessor: Any,
    model: xgb.XGBClassifier,
    required_cols: list[str],
) -> dict[str, Any]:
    raw = pd.DataFrame([input_dict])
    engineered = engineer_features(raw)
    X = engineered.reindex(columns=required_cols)
    processed = preprocessor.transform(X)
    probability = float(model.predict_proba(processed)[0, 1])
    prediction = int(probability >= BUSINESS_THRESHOLD)
    top_feature = xgboost_top_features(X, processed, preprocessor, model, required_cols)[0]
    archetype = assign_archetype(engineered.iloc[0])
    play, action, owner, urgency = recommend_play(archetype, probability)
    return {
        "probability": probability,
        "prediction": prediction,
        "tier": assign_tier(probability),
        "top_feature": top_feature,
        "archetype": archetype,
        "play": play,
        "action": action,
        "owner": owner,
        "urgency": urgency,
        "expected_exposure": probability * FN_COST,
    }


def predict_if_needed(raw_df: pd.DataFrame, preprocessor: Any, model: xgb.XGBClassifier, required_cols: list[str]) -> tuple[pd.DataFrame, str]:
    df = raw_df.copy()

    if has_predictions(df):
        df["churn_probability"] = pd.to_numeric(df["churn_probability"], errors="coerce").clip(0, 1)
        df["churn_prediction"] = pd.to_numeric(df["churn_prediction"], errors="coerce").fillna(0).astype(int)
        scored = enrich_analytics(df)
        scored = add_top_feature_column(scored, df, preprocessor, model, required_cols)
        return scored, "Uploaded file already contained churn predictions; CRIS used them directly for analytics."

    if "churn_probability" in df.columns and "churn_prediction" not in df.columns:
        df["churn_probability"] = pd.to_numeric(df["churn_probability"], errors="coerce").clip(0, 1)
        df["churn_prediction"] = (df["churn_probability"] >= BUSINESS_THRESHOLD).astype(int)
        scored = enrich_analytics(df)
        scored = add_top_feature_column(scored, df, preprocessor, model, required_cols)
        return scored, "Uploaded file had probabilities only; CRIS applied the business threshold to create predictions."

    missing = validate_for_prediction(df, required_cols)
    if missing:
        raise ValueError(
            "This file does not contain predictions and is missing model input columns: "
            + ", ".join(missing[:12])
            + ("..." if len(missing) > 12 else "")
        )

    engineered = engineer_features(df)
    X = engineered.reindex(columns=required_cols)
    processed = preprocessor.transform(X)
    probabilities = model.predict_proba(processed)[:, 1]
    engineered["churn_probability"] = probabilities
    engineered["churn_prediction"] = (probabilities >= BUSINESS_THRESHOLD).astype(int)
    scored = enrich_analytics(engineered)
    scored["top_churn_feature"] = xgboost_top_features(X, processed, preprocessor, model, required_cols)
    return scored, "Uploaded file did not contain predictions; CRIS predicted first, then generated analytics."


def assign_tier(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.40:
        return "High"
    if probability >= BUSINESS_THRESHOLD:
        return "Watch"
    return "Low"


def assign_archetype(row: pd.Series) -> str:
    if float(row.get("unresolved_complaint_count", 0) or 0) > 0 or float(row.get("escalation_count", 0) or 0) > 0:
        return "Service-Failed"
    if float(row.get("balance_decline_percentage", 0) or 0) >= 10:
        return "Financially Fleeing"
    if float(row.get("last_login_days", 0) or 0) > 21 or float(row.get("total_digital_logins", 999) or 999) < 20:
        return "Digitally Disengaged"
    if float(row.get("customer_lifetime_value", 0) or 0) > 20_000 and float(row.get("last_contacted_days", 0) or 0) > 60:
        return "Abandoned High-Value"
    return "Mixed Risk"


def recommend_play(archetype: str, probability: float) -> tuple[str, str, str, str]:
    if archetype == "Service-Failed":
        return "Rescue Clock", "Resolve complaint within 48 hours and confirm recovery", "Service ops", "48 hrs"
    if archetype == "Financially Fleeing":
        return "Flight-of-Capital Radar", "RM deposit call and targeted retention bundle", "RM / branch", "72 hrs"
    if archetype == "Digitally Disengaged":
        return "Login Lifeline", "21-day app reactivation and digital cashback", "Digital CRM", "7 days"
    if archetype == "Abandoned High-Value":
        return "Platinum Save Desk", "Priority RM outreach and portfolio review", "Priority banking", "72 hrs"
    if probability >= BUSINESS_THRESHOLD:
        return "Retention Desk Review", "Manual review for next-best action", "Retention desk", "7 days"
    return "Monitor Only", "No paid intervention; keep in monitoring queue", "CRM analytics", "Monthly"


def risk_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if float(row.get("unresolved_complaint_count", 0) or 0) > 0:
        reasons.append("unresolved complaint")
    if float(row.get("complaint_resolution_time", 0) or 0) >= 7:
        reasons.append("slow complaint resolution")
    if float(row.get("balance_decline_percentage", 0) or 0) >= 10:
        reasons.append("balance decline")
    if float(row.get("last_login_days", 0) or 0) > 21:
        reasons.append("digital inactivity")
    if float(row.get("last_contacted_days", 0) or 0) > 60:
        reasons.append("relationship contact overdue")
    if float(row.get("campaign_response_count", 1) or 1) <= 0:
        reasons.append("no campaign response")
    return "; ".join(reasons[:5]) if reasons else "mixed early-risk signals"


def enrich_analytics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["churn_probability"] = pd.to_numeric(data["churn_probability"], errors="coerce").fillna(0).clip(0, 1)
    data["churn_prediction"] = pd.to_numeric(data["churn_prediction"], errors="coerce").fillna(0).astype(int)
    data["risk_tier"] = data["churn_probability"].apply(assign_tier)
    data["expected_exposure"] = data["churn_probability"] * FN_COST
    data["intervention_cost"] = np.where(data["churn_prediction"] == 1, FP_COST, 0)
    data["net_expected_value"] = data["expected_exposure"] - data["intervention_cost"]

    if "risk_archetype" not in data.columns:
        data["risk_archetype"] = data.apply(assign_archetype, axis=1)
    if "top_churn_feature" not in data.columns:
        data["top_churn_feature"] = data.apply(strongest_business_signal, axis=1)

    plays = data.apply(lambda row: recommend_play(str(row["risk_archetype"]), float(row["churn_probability"])), axis=1)
    data["recommended_play"] = [item[0] for item in plays]
    data["next_best_action"] = [item[1] for item in plays]
    data["owner"] = [item[2] for item in plays]
    data["urgency"] = [item[3] for item in plays]
    data["risk_reason"] = data.apply(risk_reasons, axis=1)
    return data


def inr(value: float) -> str:
    value = float(value)
    if abs(value) >= 10_000_000:
        return f"INR {value / 10_000_000:.2f} Cr"
    if abs(value) >= 100_000:
        return f"INR {value / 100_000:.2f} L"
    return f"INR {value:,.0f}"


def plotly_config() -> dict[str, Any]:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


def style_fig(
    fig: go.Figure,
    height: int = 330,
    show_legend: bool | None = None,
    legend_right: bool = False,
) -> go.Figure:
    margin = dict(l=54, r=170 if legend_right else 28, t=58, b=54)
    legend = (
        dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=INK),
            title=dict(font=dict(size=11, color=MUTED)),
        )
        if legend_right
        else dict(orientation="h", y=-0.22, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED))
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Arial, Helvetica, sans-serif", color=INK, size=12),
        margin=margin,
        title=dict(x=0, xanchor="left", font=dict(size=15, color=INK)),
        legend=legend,
        hoverlabel=dict(bgcolor=PANEL_SOFT, bordercolor=RULE, font=dict(color=INK)),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    fig.update_xaxes(
        gridcolor="#EDDFCD",
        zerolinecolor=RULE,
        linecolor=RULE,
        tickfont=dict(size=10, color="#7C6C5C"),
        title_font=dict(size=11, color=MUTED),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="#EDDFCD",
        zerolinecolor=RULE,
        linecolor=RULE,
        tickfont=dict(size=10, color="#7C6C5C"),
        title_font=dict(size=11, color=MUTED),
        automargin=True,
    )
    return fig


@st.dialog("Full screen graph", width="large")
def fullscreen_chart() -> None:
    payload = st.session_state.get("fullscreen_chart")
    if not payload:
        st.info("Select a graph first.")
        return
    st.plotly_chart(
        style_fig(
            go.Figure(payload["fig"]),
            height=720,
            show_legend=payload.get("show_legend"),
            legend_right=payload.get("legend_right", False),
        ),
        use_container_width=True,
        config=plotly_config(),
    )
    st.markdown(f"<div class='chart-note'>{payload['note']}</div>", unsafe_allow_html=True)


def chart_panel(
    fig: go.Figure,
    note: str,
    key: str,
    height: int = 330,
    show_legend: bool | None = None,
    legend_right: bool = False,
) -> None:
    with st.container(border=True):
        st.plotly_chart(
            style_fig(fig, height=height, show_legend=show_legend, legend_right=legend_right),
            use_container_width=True,
            config=plotly_config(),
        )
        cols = st.columns([0.18, 0.82], vertical_alignment="center")
        with cols[0]:
            if st.button("Full screen", key=f"fullscreen_{key}", use_container_width=True):
                st.session_state["fullscreen_chart"] = {
                    "fig": go.Figure(fig),
                    "note": note,
                    "show_legend": show_legend,
                    "legend_right": legend_right,
                }
                fullscreen_chart()
        with cols[1]:
            st.caption("Hover on graph for zoom, pan, reset and download controls.")
        st.markdown(f"<div class='chart-note'>{note}</div>", unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str, str | None]], columns: int = 4) -> None:
    cols = st.columns(columns)
    for idx, (label, value, delta) in enumerate(items):
        with cols[idx % columns]:
            st.metric(label, value, delta=delta)


def warm_table(df: pd.DataFrame, max_rows: int = 100) -> None:
    display = df.head(max_rows).copy()
    html = display.to_html(index=False, escape=False, classes="warm-table")
    st.markdown(f"<div class='warm-table-wrap'>{html}</div>", unsafe_allow_html=True)


def segment_table(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    table = (
        df.groupby(col, dropna=False)
        .agg(
            customers=("churn_probability", "size"),
            avg_probability=("churn_probability", "mean"),
            flagged=("churn_prediction", "sum"),
            exposure=("expected_exposure", "sum"),
        )
        .reset_index()
        .rename(columns={col: col.replace("_", " ").title()})
        .sort_values(["exposure", "avg_probability"], ascending=False)
        .head(top_n)
    )
    table["flagged_rate"] = table["flagged"] / table["customers"]
    return table


def top_feature_frequency(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if "top_churn_feature" not in df.columns:
        return pd.DataFrame()
    counts = df["top_churn_feature"].fillna("Mixed risk signals").replace("", "Mixed risk signals").value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = pd.Series({"Other": counts.iloc[top_n:].sum()})
        counts = pd.concat([top, other])
    return pd.DataFrame({"Top feature": counts.index.astype(str), "Customers": counts.to_numpy()})


def risk_tier_insight(df: pd.DataFrame) -> str:
    actionable = int(df["risk_tier"].isin(["Critical", "High", "Watch"]).sum())
    share = actionable / max(len(df), 1)
    return f"{actionable:,} customers ({share:.1%}) need retention review; build a weekly save queue before risk converts into actual churn."


def probability_distribution_insight(df: pd.DataFrame) -> str:
    flagged = int(df["churn_prediction"].sum())
    return f"{flagged:,} customers cross the business threshold; prioritize recall because missed churn is materially costlier than excess outreach."


def segment_exposure_insight(table: pd.DataFrame) -> str:
    if table.empty:
        return "Segment exposure is unavailable; upload raw business columns to identify where retention capacity should be allocated first."
    label_col = table.columns[0]
    top = table.sort_values("exposure", ascending=False).iloc[0]
    return f"{top[label_col]} carries the highest exposure; assign focused retention capacity there before broad campaign spending."


def feature_frequency_insight(table: pd.DataFrame) -> str:
    if table.empty:
        return "Feature contribution requires raw model inputs; upload full portfolio data to identify recurring churn triggers."
    top = table.iloc[0]
    share = int(top["Customers"]) / max(int(table["Customers"].sum()), 1)
    return f"{top['Top feature']} is the most frequent risk driver ({share:.1%}); convert it into a monitored operating trigger."


def balance_decline_insight(df: pd.DataFrame) -> str:
    high_decline = df[pd.to_numeric(df["balance_decline_percentage"], errors="coerce").fillna(0) >= 10]
    if high_decline.empty:
        return "Balance decline is not concentrated here; keep monitoring silent money movement before it becomes visible attrition."
    return f"{len(high_decline):,} customers show 10%+ balance decline; trigger RM outreach before funds fully leave the bank."


def digital_login_insight(df: pd.DataFrame) -> str:
    low_login = df[pd.to_numeric(df["total_digital_logins"], errors="coerce").fillna(999) < 20]
    if low_login.empty:
        return "Digital activity is healthy overall; maintain light engagement nudges instead of expensive retention offers."
    return f"{len(low_login):,} customers have low digital logins; run reactivation nudges before disengagement becomes account closure."


def render_upload(required_cols: list[str], preprocessor: Any, model: xgb.XGBClassifier) -> pd.DataFrame | None:
    st.subheader("Upload portfolio")
    st.markdown(
        "<div class='note'>Upload a CSV to unlock analytics. The app does not load or expose private data by default.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is None:
        st.markdown(
            """
            <div class='callout'>
            <b>No dataset loaded.</b><br>
            Upload either a raw customer portfolio or a scored file. If the file already has
            <code>churn_prediction</code> and <code>churn_probability</code>, CRIS will analyze it directly.
            If not, CRIS will score the file first using the bundled model artifacts.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None

    raw_df = pd.read_csv(uploaded)
    try:
        scored, message = predict_if_needed(raw_df, preprocessor, model, required_cols)
    except Exception as exc:
        st.error(str(exc))
        return None

    st.session_state["scored"] = scored
    st.session_state["source_name"] = uploaded.name
    st.success(message)

    prediction_output = scored[["customer_id", "churn_prediction", "churn_probability"]].copy() if "customer_id" in scored.columns else scored[["churn_prediction", "churn_probability"]].copy()
    st.download_button(
        "Download prediction CSV",
        prediction_output.to_csv(index=False).encode("utf-8"),
        "ChurnZero_TeamName_Predictions.csv",
        "text/csv",
        use_container_width=True,
    )
    return scored


def render_summary(scored: pd.DataFrame) -> None:
    customers = len(scored)
    flagged = int(scored["churn_prediction"].sum())
    avg_prob = float(scored["churn_probability"].mean())
    exposure = float(scored["expected_exposure"].sum())
    cost = float(scored["intervention_cost"].sum())
    net_value = exposure - cost
    top_play = scored.loc[scored["churn_prediction"] == 1, "recommended_play"].value_counts()
    top_play_text = top_play.index[0] if len(top_play) else "Monitor Only"

    st.subheader("Executive business summary")
    metric_row(
        [
            ("Customers analyzed", f"{customers:,}", None),
            ("Flagged for retention", f"{flagged:,}", f"{flagged / max(customers, 1):.1%} of uploaded file"),
            ("Average churn probability", f"{avg_prob:.1%}", None),
            ("Expected churn exposure", inr(exposure), None),
            ("Intervention cost", inr(cost), f"INR {FP_COST:,} per flagged customer"),
            ("Net expected value", inr(net_value), None),
            ("Business threshold", f"{BUSINESS_THRESHOLD:.3f}", "not default 0.5"),
            ("Top response play", top_play_text, None),
        ],
        columns=4,
    )

    st.markdown(
        f"""
        <div class='callout'>
        CRIS analyzed <b>{customers:,}</b> uploaded customers and identified <b>{flagged:,}</b>
        customers above the intervention threshold. The expected churn exposure is <b>{inr(exposure)}</b>,
        while targeted interventions cost <b>{inr(cost)}</b>. Priority response: <b>{top_play_text}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(scored: pd.DataFrame) -> None:
    st.subheader("Portfolio analytics")

    c1, c2 = st.columns(2, gap="medium", vertical_alignment="top")
    with c1:
        tier_counts = scored["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk tier", "Customers"]
        fig = px.pie(
            tier_counts,
            names="Risk tier",
            values="Customers",
            hole=0.55,
            title="Risk tier split",
            color="Risk tier",
            color_discrete_map={"Critical": RUST, "High": AMBER, "Watch": CLAY, "Low": SAGE},
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
            marker=dict(line=dict(color=PANEL, width=2)),
        )
        fig.update_layout(legend_title_text="Risk tier")
        chart_panel(fig, risk_tier_insight(scored), "risk_tier", show_legend=True, legend_right=True)

    with c2:
        fig = px.histogram(scored, x="churn_probability", nbins=35, title="Churn probability distribution", color_discrete_sequence=[RUST])
        fig.add_vline(x=BUSINESS_THRESHOLD, line_color=AMBER, line_dash="dash")
        chart_panel(fig, probability_distribution_insight(scored), "probability_distribution", show_legend=False)

    st.markdown("#### Segment exposure")
    segment_cols = [c for c in ["card_category", "primary_account_type", "region", "city_tier", "income_band", "customer_segment"] if c in scored.columns]
    if segment_cols:
        chosen = st.selectbox("Segment view", segment_cols)
        table = segment_table(scored, chosen)
        display = table.copy()
        if not display.empty:
            display["avg_probability"] = display["avg_probability"].map(lambda x: f"{x:.1%}")
            display["flagged_rate"] = display["flagged_rate"].map(lambda x: f"{x:.1%}")
            display["exposure"] = display["exposure"].map(inr)
            warm_table(display)

            plot_table = table.sort_values("exposure")
            y_col = plot_table.columns[0]
            fig = px.bar(plot_table, x="exposure", y=y_col, orientation="h", title=f"Expected exposure by {y_col}", color_discrete_sequence=[RUST])
            chart_panel(fig, segment_exposure_insight(table), f"segment_{chosen}", show_legend=False)
    else:
        st.info("Segment charts require business feature columns such as card_category, account type, region or income band.")

    st.markdown("#### Feature contribution frequency")
    feature_freq = top_feature_frequency(scored)
    if not feature_freq.empty:
        fig = px.pie(
            feature_freq,
            names="Top feature",
            values="Customers",
            hole=0.48,
            title="Frequency of top churn features",
            color_discrete_sequence=[RUST, AMBER, SAGE, TERRACOTTA, CLAY, "#8E735B", "#BFA184", "#6F8A72", "#D7C3A8"],
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
            marker=dict(line=dict(color=PANEL, width=2)),
        )
        fig.update_layout(legend_title_text="Top feature")
        chart_panel(
            fig,
            feature_frequency_insight(feature_freq),
            "top_feature_frequency",
            height=360,
            show_legend=True,
            legend_right=True,
        )
    else:
        st.info("Top feature frequency is unavailable because the uploaded file does not include enough model inputs.")

    st.markdown("#### Driver evidence")
    d1, d2 = st.columns(2, gap="medium", vertical_alignment="top")
    with d1:
        if "balance_decline_percentage" in scored.columns:
            fig = px.scatter(
                scored,
                x="balance_decline_percentage",
                y="churn_probability",
                color="risk_tier",
                color_discrete_map={"Critical": RUST, "High": AMBER, "Watch": CLAY, "Low": SAGE},
                hover_data=[c for c in ["customer_id", "risk_archetype"] if c in scored.columns],
                title="Balance decline vs churn probability",
            )
            chart_panel(fig, balance_decline_insight(scored), "balance_decline", height=350)
        else:
            st.info("Balance decline chart unavailable because balance_decline_percentage is not in the uploaded file.")

    with d2:
        if "total_digital_logins" in scored.columns:
            fig = px.scatter(
                scored,
                x="total_digital_logins",
                y="churn_probability",
                color="risk_tier",
                color_discrete_map={"Critical": RUST, "High": AMBER, "Watch": CLAY, "Low": SAGE},
                hover_data=[c for c in ["customer_id", "recommended_play"] if c in scored.columns],
                title="Digital logins vs churn probability",
            )
            chart_panel(fig, digital_login_insight(scored), "digital_logins", height=350)
        else:
            st.info("Digital engagement chart unavailable because total_digital_logins is not in the uploaded file.")


def render_customer_and_playbook(scored: pd.DataFrame) -> None:
    st.subheader("Customer profiler and playbook")
    if "customer_id" in scored.columns:
        ids = scored.sort_values("churn_probability", ascending=False)["customer_id"].astype(str).tolist()
        selected = st.selectbox("Select customer ID", ids)
        row = scored.loc[scored["customer_id"].astype(str) == selected].iloc[0]
    else:
        row = scored.sort_values("churn_probability", ascending=False).iloc[0]

    metric_row(
        [
            ("Churn probability", f"{float(row['churn_probability']):.1%}", row["risk_tier"]),
            ("Expected exposure", inr(float(row["expected_exposure"])), None),
            ("Archetype", str(row["risk_archetype"]), str(row["recommended_play"])),
            ("Top churn feature", str(row.get("top_churn_feature", "Mixed risk signals")), None),
            ("Owner", str(row["owner"]), str(row["urgency"])),
        ],
        columns=5,
    )

    st.markdown(
        f"<div class='callout'><b>Top churn feature:</b> {row.get('top_churn_feature', 'Mixed risk signals')}<br><b>Risk reasons:</b> {row['risk_reason']}<br><b>Action:</b> {row['next_best_action']}</div>",
        unsafe_allow_html=True,
    )

    queue = scored[scored["churn_prediction"] == 1].sort_values("expected_exposure", ascending=False)
    cols = [
        c
        for c in [
            "customer_id",
            "risk_tier",
            "risk_archetype",
            "top_churn_feature",
            "churn_probability",
            "expected_exposure",
            "recommended_play",
            "owner",
            "urgency",
            "next_best_action",
        ]
        if c in queue.columns
    ]
    warm_table(queue[cols], max_rows=100)
    st.download_button(
        "Download retention playbook",
        queue[cols + ["risk_reason"]].to_csv(index=False).encode("utf-8"),
        "CRIS_Retention_Playbook.csv",
        "text/csv",
        use_container_width=True,
    )


def numeric_driver_input(schema: dict[str, Any], feature: str, label: str, key: str) -> Any:
    _, _, default = schema_numeric_bounds(schema, feature)
    integer_like = feature.endswith("_count") or feature in {"tenure_months", "total_digital_logins"}
    if integer_like:
        return st.number_input(
            label,
            value=int(round(default)),
            step=1,
            key=key,
        )
    return st.number_input(
        label,
        value=float(default),
        step=1.0,
        format="%.2f",
        key=key,
    )


def categorical_driver_input(schema: dict[str, Any], feature: str, label: str, key: str) -> Any:
    details = schema.get(feature, {})
    options = details.get("options", [])
    default = details.get("default")
    if default not in options and options:
        default = options[0]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options=options or [default], index=index, key=key)


def render_top10_driver_simulator(
    schema: dict[str, Any],
    preprocessor: Any,
    model: xgb.XGBClassifier,
    required_cols: list[str],
) -> None:
    st.subheader("Top 10 primary churn drivers")
    st.markdown(
        "<div class='note'>Tune the most important business-facing drivers and calculate one customer's churn probability using the same XGBoost model.</div>",
        unsafe_allow_html=True,
    )

    input_dict = schema_default_row(schema)
    col1, col2, col3 = st.columns(3, gap="medium", vertical_alignment="top")

    with col1:
        input_dict["unresolved_complaint_count"] = numeric_driver_input(
            schema, "unresolved_complaint_count", "Unresolved Complaints", "driver_unresolved"
        )
        input_dict["complaint_resolution_time"] = numeric_driver_input(
            schema, "complaint_resolution_time", "Resolution Time (Days)", "driver_resolution"
        )
        _, _, default_decline = schema_numeric_bounds(schema, "balance_decline_percentage")
        input_dict["balance_decline_percentage"] = st.slider(
            "Balance Decline Percentage",
            min_value=0.0,
            max_value=100.0,
            value=float(np.clip(default_decline, 0, 100)),
            step=0.5,
            key="driver_balance_decline",
        )
        input_dict["total_digital_logins"] = numeric_driver_input(
            schema, "total_digital_logins", "Total Digital Logins", "driver_total_logins"
        )

    with col2:
        input_dict["tenure_months"] = numeric_driver_input(schema, "tenure_months", "Tenure (Months)", "driver_tenure")
        input_dict["monthly_transaction_count"] = numeric_driver_input(
            schema, "monthly_transaction_count", "Monthly Transactions", "driver_monthly_txn"
        )
        input_dict["avg_monthly_balance"] = numeric_driver_input(
            schema, "avg_monthly_balance", "Avg Monthly Balance", "driver_avg_balance"
        )
        input_dict["annual_income"] = numeric_driver_input(schema, "annual_income", "Annual Income", "driver_income")

    with col3:
        input_dict["education_level"] = categorical_driver_input(
            schema, "education_level", "Education Level", "driver_education"
        )
        input_dict["city_tier"] = categorical_driver_input(schema, "city_tier", "City Tier", "driver_city_tier")

    st.markdown(
        """
        <div class='callout'>
        The remaining features are pre-filled using the saved schema defaults from the training pipeline.
        This simulator is directional: use it to understand risk movement, then validate with portfolio-level scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Calculate churn risk", type="primary", use_container_width=True):
        result = score_single_customer(input_dict, preprocessor, model, required_cols)
        metric_row(
            [
                ("Calculated churn probability", f"{result['probability']:.2%}", result["tier"]),
                ("Prediction", "Churn risk" if result["prediction"] else "Low risk", f"threshold {BUSINESS_THRESHOLD:.3f}"),
                ("Top churn feature", str(result["top_feature"]), None),
                ("Expected exposure", inr(result["expected_exposure"]), None),
            ],
            columns=4,
        )
        st.progress(float(result["probability"]))
        st.markdown(
            f"""
            <div class='callout'>
            <b>Recommended play:</b> {result['play']}<br>
            <b>Action:</b> {result['action']}<br>
            <b>Owner / urgency:</b> {result['owner']} / {result['urgency']}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    apply_style()
    preprocessor, model, schema = load_assets()
    required_cols = expected_model_columns(preprocessor, schema)

    st.markdown(
        """
        <div class='hero'>
            <div class='hero-kicker'>Customer Risk Intelligence System</div>
            <h1>CRIS turns uploaded churn data into retention ROI.</h1>
            <div class='hero-subtitle'>
                Upload a raw customer portfolio to predict churn first, or upload an already-scored file
                to go directly into analytics. No private dataset is loaded by default.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### CRIS")
        st.caption("Upload-first, GitHub-safe dashboard")
        st.divider()
        st.caption(f"Business threshold: {BUSINESS_THRESHOLD:.3f}")
        st.caption(f"False negative cost: INR {FN_COST:,}")
        st.caption(f"False positive cost: INR {FP_COST:,}")
        st.divider()
        st.caption("Private CSV files should not be committed to GitHub.")

    scored = render_upload(required_cols, preprocessor, model)
    if scored is None:
        return

    tabs = st.tabs(["Executive Summary", "Portfolio Analytics", "Customer + Playbook", "Top 10 Drivers"])
    with tabs[0]:
        render_summary(scored)
    with tabs[1]:
        render_dashboard(scored)
    with tabs[2]:
        render_customer_and_playbook(scored)
    with tabs[3]:
        render_top10_driver_simulator(schema, preprocessor, model, required_cols)


if __name__ == "__main__":
    main()
