"""
Heart Disease Prediction System — Premium Healthcare AI Dashboard

A modern Streamlit application for cardiovascular risk assessment
with glassmorphism UI, interactive Plotly analytics, and ML predictions.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction | AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

NAV_ITEMS = [
    ("dashboard", "🏠 Dashboard"),
    ("prediction", "🩺 Prediction"),
    ("analytics", "📈 Analytics"),
    ("history", "📋 Patient History"),
    ("insights", "🧠 Model Insights"),
    ("dataset", "📊 Dataset Analysis"),
    ("about", "ℹ About"),
]

FEATURE_CONFIG = {
    "age": {
        "label": "Age",
        "icon": "🎂",
        "type": "number",
        "min": 1,
        "max": 120,
        "default": 54,
        "help": "Age of the patient in years",
        "group": "demographics",
    },
    "sex": {
        "label": "Sex",
        "icon": "👤",
        "type": "select",
        "options": {0: "Female", 1: "Male"},
        "default": 1,
        "help": "Biological sex of the patient",
        "group": "demographics",
    },
    "cp": {
        "label": "Chest Pain Type",
        "icon": "💔",
        "type": "select",
        "options": {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic",
        },
        "default": 0,
        "help": "Type of chest pain experienced",
        "group": "demographics",
    },
    "trestbps": {
        "label": "Blood Pressure",
        "icon": "🩸",
        "type": "number",
        "min": 80,
        "max": 250,
        "default": 120,
        "help": "Resting blood pressure in mm Hg",
        "group": "vitals",
    },
    "chol": {
        "label": "Cholesterol",
        "icon": "🧪",
        "type": "number",
        "min": 100,
        "max": 600,
        "default": 200,
        "help": "Serum cholesterol in mg/dl",
        "group": "vitals",
    },
    "thalach": {
        "label": "Heart Rate",
        "icon": "💓",
        "type": "number",
        "min": 60,
        "max": 220,
        "default": 150,
        "help": "Maximum heart rate achieved",
        "group": "vitals",
    },
    "fbs": {
        "label": "Fasting Blood Sugar",
        "icon": "🍬",
        "type": "select",
        "options": {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
        "default": 0,
        "help": "Fasting blood sugar > 120 mg/dl",
        "group": "clinical",
    },
    "exang": {
        "label": "Exercise Angina",
        "icon": "🏃",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
        "default": 0,
        "help": "Exercise induced chest pain",
        "group": "clinical",
    },
    "restecg": {
        "label": "Rest ECG",
        "icon": "📈",
        "type": "select",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy",
        },
        "default": 0,
        "help": "Resting electrocardiographic results",
        "group": "clinical",
    },
    "oldpeak": {
        "label": "Old Peak",
        "icon": "📉",
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "default": 1.0,
        "step": 0.1,
        "help": "ST depression induced by exercise relative to rest",
        "group": "clinical",
    },
    "slope": {
        "label": "Slope",
        "icon": "📐",
        "type": "select",
        "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "default": 1,
        "help": "Slope of the peak exercise ST segment",
        "group": "clinical",
    },
    "ca": {
        "label": "CA (Major Vessels)",
        "icon": "🫀",
        "type": "select",
        "options": {0: "0", 1: "1", 2: "2", 3: "3"},
        "default": 0,
        "help": "Number of major vessels colored by fluoroscopy",
        "group": "clinical",
    },
    "thal": {
        "label": "Thal",
        "icon": "🔬",
        "type": "select",
        "options": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"},
        "default": 2,
        "help": "Thalassemia blood disorder type",
        "group": "clinical",
    },
}

INPUT_GROUPS = {
    "demographics": ("👥 Demographics", ["age", "sex", "cp"]),
    "vitals": ("🩺 Vital Parameters", ["trestbps", "chol", "thalach"]),
    "clinical": (
        "🔬 Clinical Parameters",
        ["fbs", "exang", "restecg", "oldpeak", "slope", "ca", "thal"],
    ),
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#e2e8f0", family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "page": "dashboard",
        "prediction_history": [],
        "last_prediction": None,
        "form_reset_counter": 0,
        "predict_clicked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Custom CSS — Dark glassmorphism theme
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}

        .stApp {
            background: radial-gradient(ellipse at 20% 0%, #1e3a5f 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 100%, #4c1d95 0%, transparent 50%),
                        linear-gradient(160deg, #0f172a 0%, #1e1b4b 45%, #0f172a 100%);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(30,27,75,0.92) 100%) !important;
            border-right: 1px solid rgba(148,163,184,0.12);
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1400px;
        }

        .top-header {
            background: linear-gradient(135deg, rgba(30,41,59,0.75), rgba(49,46,129,0.55));
            backdrop-filter: blur(16px);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        }
        .top-header h1 {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #f472b6, #fb7185, #fda4af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .top-header .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin: 0.35rem 0 0;
        }
        .header-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .header-chip {
            background: rgba(15,23,42,0.6);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 12px;
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        .header-chip strong { color: #f8fafc; }

        .hero-section {
            background: linear-gradient(135deg, rgba(225,29,72,0.15), rgba(79,70,229,0.2));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(244,114,182,0.25);
            border-radius: 24px;
            padding: 2.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            text-align: center;
        }
        .hero-section h1 {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            color: #f8fafc;
        }
        .hero-section p {
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0.75rem 0 0;
        }

        .glass-card {
            background: rgba(30,41,59,0.55);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        }
        .glass-card h3 {
            color: #f1f5f9;
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 0.5rem;
        }

        .kpi-card {
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
        }
        .kpi-card:hover { transform: scale(1.03); }
        .kpi-card .kpi-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.85;
            margin-bottom: 0.35rem;
        }
        .kpi-card .kpi-value {
            font-size: 1.75rem;
            font-weight: 800;
        }
        .kpi-blue { background: linear-gradient(135deg, #1d4ed8, #3b82f6); color: white; }
        .kpi-purple { background: linear-gradient(135deg, #6d28d9, #8b5cf6); color: white; }
        .kpi-green { background: linear-gradient(135deg, #047857, #10b981); color: white; }
        .kpi-orange { background: linear-gradient(135deg, #c2410c, #f97316); color: white; }
        .kpi-red { background: linear-gradient(135deg, #b91c1c, #ef4444); color: white; }
        .kpi-teal { background: linear-gradient(135deg, #0f766e, #14b8a6); color: white; }

        .risk-low {
            background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(52,211,153,0.15));
            border: 2px solid #10b981;
            color: #ecfdf5;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            box-shadow: 0 0 40px rgba(16,185,129,0.25);
            animation: pulse-green 2s infinite;
        }
        .risk-high {
            background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(248,113,113,0.15));
            border: 2px solid #ef4444;
            color: #fef2f2;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            box-shadow: 0 0 40px rgba(239,68,68,0.25);
            animation: pulse-red 2s infinite;
        }
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.2); }
            50% { box-shadow: 0 0 40px rgba(16,185,129,0.45); }
        }
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.2); }
            50% { box-shadow: 0 0 40px rgba(239,68,68,0.45); }
        }

        .sidebar-logo {
            text-align: center;
            padding: 1rem 0 0.5rem;
        }
        .sidebar-logo .logo-icon { font-size: 2.5rem; }
        .sidebar-logo .logo-text {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 0.25rem;
        }
        .sidebar-logo .logo-sub {
            font-size: 0.75rem;
            color: #64748b;
        }

        .progress-bar-wrap {
            background: rgba(15,23,42,0.6);
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s ease;
        }

        .success-card {
            background: rgba(16,185,129,0.15);
            border-left: 4px solid #10b981;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            color: #d1fae5;
            margin: 0.75rem 0;
        }
        .warning-card {
            background: rgba(245,158,11,0.15);
            border-left: 4px solid #f59e0b;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            color: #fef3c7;
            margin: 0.75rem 0;
        }

        div[data-testid="stExpander"] {
            background: rgba(30,41,59,0.45);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 14px;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #e11d48, #be123c) !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            box-shadow: 0 6px 24px rgba(225,29,72,0.45) !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 32px rgba(225,29,72,0.55) !important;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(244,114,182,0.3);
        }

        .input-card-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# HTML Helpers
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, color: str) -> str:
    return f"""
    <div class="kpi-card kpi-{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


def progress_bar_html(pct: float, color: str) -> str:
    return f"""
    <div class="progress-bar-wrap">
        <div class="progress-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


def render_top_header(metadata: dict):
    metrics = metadata.get("metrics", {})
    best_name = metadata.get("best_model_name", "N/A")
    accuracy = metrics.get("Accuracy", 0)
    roc_auc = metrics.get("ROC-AUC", 0)
    today = datetime.now().strftime("%B %d, %Y")

    st.markdown(
        f"""
        <div class="top-header">
            <h1>❤️ Heart Disease Prediction System</h1>
            <p class="subtitle">AI-Powered Cardiovascular Risk Assessment</p>
            <div class="header-meta">
                <span class="header-chip">🎯 <strong>Accuracy:</strong> {accuracy:.1%}</span>
                <span class="header-chip">📊 <strong>ROC-AUC:</strong> {roc_auc:.4f}</span>
                <span class="header-chip">🤖 <strong>Model:</strong> {best_name}</span>
                <span class="header-chip">📅 <strong>Date:</strong> {today}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return None, None, None
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
        return model, scaler, metadata
    except Exception as exc:
        st.error(f"Error loading model artifacts: {exc}")
        return None, None, None


@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates()
    return df


@st.cache_data
def get_evaluation_data(_model, _scaler, feature_names):
    df = load_dataset()
    if df is None:
        return None, None, None, None
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_scaled = _scaler.transform(X_test[feature_names])
    y_prob = _model.predict_proba(X_scaled)[:, 1]
    y_pred = _model.predict(X_scaled)
    return y_test.values, y_pred, y_prob, X_test


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})
    df = pd.DataFrame({"feature": feature_names, "importance": values})
    return df.sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# Input Widgets
# ---------------------------------------------------------------------------
def build_input_widget(feature_key: str, config: dict):
    label = f"{config.get('icon', '')} {config['label']}"
    help_text = config.get("help", "")
    reset_key = st.session_state.form_reset_counter
    widget_key = f"input_{feature_key}_{reset_key}"

    if config["type"] == "select":
        options = list(config["options"].keys())
        labels = [config["options"][k] for k in options]
        default_idx = options.index(config["default"])
        selected_label = st.selectbox(
            label, labels, index=default_idx, help=help_text, key=widget_key
        )
        return options[labels.index(selected_label)]

    return st.number_input(
        label,
        min_value=config.get("min", 0),
        max_value=config.get("max", 999),
        value=config["default"],
        step=config.get("step", 1),
        help=help_text,
        key=widget_key,
    )


def collect_user_inputs():
    inputs = {}
    for key, cfg in FEATURE_CONFIG.items():
        inputs[key] = build_input_widget(key, cfg)
    return inputs


def run_prediction(model, scaler, metadata, user_inputs):
    feature_names = metadata.get("feature_names", list(FEATURE_CONFIG.keys()))
    input_df = pd.DataFrame([user_inputs])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    disease_prob = float(probabilities[1])
    confidence = float(max(probabilities)) * 100
    metrics = metadata.get("metrics", {})
    return {
        "prediction": prediction,
        "disease_prob": disease_prob,
        "confidence": confidence,
        "probabilities": probabilities,
        "inputs": user_inputs,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_accuracy": metrics.get("Accuracy", 0) * 100,
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
    }


# ---------------------------------------------------------------------------
# Plotly Charts
# ---------------------------------------------------------------------------
def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.15)")
    return fig


def create_risk_gauge(probability: float):
    pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
            title={"text": "Risk Meter", "font": {"size": 20, "color": "#e2e8f0"}},
            delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#f472b6", "thickness": 0.25},
                "bgcolor": "rgba(15,23,42,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(16,185,129,0.5)"},
                    {"range": [25, 50], "color": "rgba(234,179,8,0.5)"},
                    {"range": [50, 75], "color": "rgba(249,115,22,0.5)"},
                    {"range": [75, 100], "color": "rgba(239,68,68,0.5)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(height=320)
    return apply_plotly_theme(fig)


def create_disease_pie(df):
    counts = df["target"].value_counts()
    labels = ["No Disease", "Heart Disease"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=["#10b981", "#ef4444"], line=dict(color="#0f172a", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#f8fafc"),
        )
    )
    fig.update_layout(title="Disease vs No Disease", height=380)
    return apply_plotly_theme(fig)


def create_feature_importance_chart(model, feature_names):
    imp_df = get_feature_importance(model, feature_names)
    fig = px.bar(
        imp_df.head(13),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#6366f1", "#ec4899"],
        title="Feature Importance",
    )
    fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    fig.update_yaxes(categoryorder="total ascending")
    return apply_plotly_theme(fig)


def create_age_histogram(df):
    fig = px.histogram(
        df, x="age", color="target",
        nbins=25,
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        labels={"age": "Age", "target": "Diagnosis", "count": "Count"},
        title="Age Distribution",
        barmode="overlay",
        opacity=0.75,
    )
    fig.update_layout(height=380)
    return apply_plotly_theme(fig)


def create_age_chol_scatter(df):
    fig = px.scatter(
        df, x="age", y="chol", color="target",
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        labels={"age": "Age", "chol": "Cholesterol", "target": "Diagnosis"},
        title="Age vs Cholesterol",
        opacity=0.7,
        size_max=12,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#0f172a")))
    fig.update_layout(height=380)
    return apply_plotly_theme(fig)


def create_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(title="Correlation Matrix", height=480)
    return apply_plotly_theme(fig)


def create_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.4f})",
                             line=dict(color="#818cf8", width=3), fill="tozeroy",
                             fillcolor="rgba(129,140,248,0.15)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return apply_plotly_theme(fig)


def create_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{cm[i][j]}<br>{labels[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#1e293b"], [1, "#6366f1"]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#f8fafc"},
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix", height=380)
    return apply_plotly_theme(fig)


def create_probability_bar(prob_no, prob_yes):
    fig = go.Figure(
        go.Bar(
            x=[prob_no * 100, prob_yes * 100],
            y=["No Disease", "Heart Disease"],
            orientation="h",
            marker_color=["#10b981", "#ef4444"],
            text=[f"{prob_no * 100:.1f}%", f"{prob_yes * 100:.1f}%"],
            textposition="auto",
            textfont=dict(color="#f8fafc"),
        )
    )
    fig.update_layout(title="Class Probabilities", xaxis_title="Probability (%)", height=280)
    return apply_plotly_theme(fig)


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------
def _pdf_safe_text(text) -> str:
    return str(text).replace("≤", "<=").replace("—", "-").replace("⚠", "").replace("✅", "")


def generate_pdf_report(result: dict, metadata: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Heart Disease Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {result['timestamp']}", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Risk Level: {result['risk_label']}", ln=True)
    pdf.cell(0, 7, f"Disease Probability: {result['disease_prob'] * 100:.1f}%", ln=True)
    pdf.cell(0, 7, f"Confidence: {result['confidence']:.1f}%", ln=True)
    pdf.cell(0, 7, f"Model: {metadata.get('best_model_name', 'N/A')}", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient Inputs", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for key, val in result["inputs"].items():
        cfg = FEATURE_CONFIG[key]
        display = cfg["options"].get(val, val) if cfg["type"] == "select" else val
        pdf.cell(0, 6, _pdf_safe_text(f"{cfg['label']}: {display}"), ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: For educational purposes only. Not a substitute for professional medical advice.",
    )
    return pdf.output()


def inputs_to_display_df(user_inputs):
    rows = []
    for key, val in user_inputs.items():
        cfg = FEATURE_CONFIG[key]
        display = cfg["options"].get(val, val) if cfg["type"] == "select" else val
        rows.append({"Feature": cfg["label"], "Value": display})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(metadata: dict):
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-icon">❤️</div>
            <div class="logo-text">CardioAI</div>
            <div class="logo-sub">Heart Disease Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    labels = [item[1] for item in NAV_ITEMS]
    keys = [item[0] for item in NAV_ITEMS]
    current_idx = keys.index(st.session_state.page) if st.session_state.page in keys else 0

    selected = st.sidebar.radio(
        "Navigation",
        labels,
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.page = keys[labels.index(selected)]

    st.sidebar.markdown("---")
    metrics = metadata.get("metrics", {})
    st.sidebar.markdown("##### 📌 Quick Stats")
    st.sidebar.metric("Model Accuracy", f"{metrics.get('Accuracy', 0):.1%}")
    st.sidebar.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}")
    st.sidebar.markdown(
        f"<small style='color:#64748b;'>Predictions: {len(st.session_state.prediction_history)}</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_dashboard(metadata: dict, df):
    metrics = metadata.get("metrics", {})
    st.markdown(
        """
        <div class="hero-section">
            <h1>❤️ Heart Disease Prediction System</h1>
            <p>AI-powered prediction using machine learning for cardiovascular risk assessment</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Accuracy", f"{metrics.get('Accuracy', 0):.1%}", "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}", "purple"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Precision", f"{metrics.get('Precision', 0):.1%}", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Recall", f"{metrics.get('Recall', 0):.1%}", "orange"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Overview</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if df is not None:
            st.plotly_chart(create_disease_pie(df), use_container_width=True)
    with col_b:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🚀 Platform Highlights</h3>
                <p style="color:#94a3b8; line-height:1.8;">
                • Real-time ML-powered risk scoring<br>
                • Interactive Plotly analytics dashboard<br>
                • Multi-model comparison & evaluation<br>
                • Patient history & exportable reports<br>
                • Enterprise-grade dark UI design
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        best = metadata.get("best_model_name", "N/A")
        st.markdown(
            f"""
            <div class="success-card">
                ✅ Best performing model: <strong>{best}</strong> selected by ROC-AUC score
            </div>
            """,
            unsafe_allow_html=True,
        )
        if df is not None:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h3>📁 Dataset Snapshot</h3>
                    <p style="color:#94a3b8;">{len(df)} records · {df.shape[1]} columns ·
                    {df['target'].sum()} positive cases</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.last_prediction:
        st.markdown('<div class="section-title">🔮 Latest Prediction</div>', unsafe_allow_html=True)
        res = st.session_state.last_prediction
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.markdown(kpi_card("Probability", f"{res['disease_prob']*100:.1f}%", "red"), unsafe_allow_html=True)
        with lc2:
            st.markdown(kpi_card("Confidence", f"{res['confidence']:.1f}%", "teal"), unsafe_allow_html=True)
        with lc3:
            st.markdown(kpi_card("Accuracy", f"{res['model_accuracy']:.1f}%", "blue"), unsafe_allow_html=True)
        with lc4:
            color = "red" if res["prediction"] == 1 else "green"
            st.markdown(kpi_card("Risk", res["risk_label"], color), unsafe_allow_html=True)


def render_prediction_page(model, scaler, metadata):
    st.markdown('<div class="section-title">🩺 Patient Risk Assessment</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    groups = [col1, col2, col3]
    group_keys = ["demographics", "vitals", "clinical"]

    user_inputs = {}
    for col, gkey in zip(groups, group_keys):
        title, features = INPUT_GROUPS[gkey]
        with col:
            with st.expander(title, expanded=True):
                st.markdown(f'<div class="input-card-title">{title}</div>', unsafe_allow_html=True)
                for feat in features:
                    cfg = FEATURE_CONFIG[feat]
                    user_inputs[feat] = build_input_widget(feat, cfg)

    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        predict = st.button("❤️ Predict Risk", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.form_reset_counter += 1
            st.session_state.last_prediction = None
            st.rerun()
    with btn_col3:
        st.markdown(
            '<div class="warning-card">⚠️ Educational use only</div>',
            unsafe_allow_html=True,
        )

    if predict:
        with st.spinner("🔄 Analyzing cardiovascular risk..."):
            import time
            time.sleep(0.8)
            result = run_prediction(model, scaler, metadata, user_inputs)
            st.session_state.last_prediction = result
            st.session_state.prediction_history.insert(0, result)

    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1])
        with r1:
            if result["prediction"] == 0:
                st.markdown('<div class="risk-low">✅ Low Risk — No Heart Disease Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-high">⚠️ High Risk — Heart Disease Likely</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="glass-card" style="margin-top:1rem;">
                    <h3>📊 Risk Breakdown</h3>
                    <p style="color:#94a3b8;">Probability: <strong style="color:#f8fafc;">{result['disease_prob']*100:.1f}%</strong></p>
                    {progress_bar_html(result['disease_prob']*100, '#ef4444')}
                    <p style="color:#94a3b8;">Confidence: <strong style="color:#f8fafc;">{result['confidence']:.1f}%</strong></p>
                    {progress_bar_html(result['confidence'], '#10b981')}
                    <p style="color:#94a3b8;">Risk Level: <strong style="color:#f8fafc;">{result['risk_label']}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.plotly_chart(create_risk_gauge(result["disease_prob"]), use_container_width=True)

        st.markdown('<div class="section-title">📈 Result Dashboard</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(kpi_card("Probability", f"{result['disease_prob']*100:.1f}%", "red"), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_card("Confidence", f"{result['confidence']:.1f}%", "teal"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_card("Model Accuracy", f"{result['model_accuracy']:.1f}%", "blue"), unsafe_allow_html=True)
        with k4:
            color = "red" if result["prediction"] == 1 else "green"
            st.markdown(kpi_card("Risk Category", result["risk_label"], color), unsafe_allow_html=True)

        st.plotly_chart(
            create_probability_bar(result["probabilities"][0], result["probabilities"][1]),
            use_container_width=True,
        )

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("📋 Input Summary", expanded=False):
                st.dataframe(inputs_to_display_df(result["inputs"]), use_container_width=True, hide_index=True)
        with exp_col2:
            with st.expander("💾 Export Options", expanded=False):
                pdf_bytes = generate_pdf_report(result, metadata)
                st.download_button(
                    "📄 Download PDF Report",
                    data=bytes(pdf_bytes),
                    file_name=f"heart_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                csv_df = pd.DataFrame([{
                    **{FEATURE_CONFIG[k]["label"]: FEATURE_CONFIG[k]["options"].get(v, v)
                       if FEATURE_CONFIG[k]["type"] == "select" else v
                       for k, v in result["inputs"].items()},
                    "Risk Level": result["risk_label"],
                    "Probability (%)": round(result["disease_prob"] * 100, 2),
                    "Confidence (%)": round(result["confidence"], 2),
                    "Timestamp": result["timestamp"],
                }])
                st.download_button(
                    "📊 Download CSV",
                    data=csv_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"heart_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def render_analytics_page(model, scaler, metadata, df):
    st.markdown('<div class="section-title">📈 Visual Analytics</div>', unsafe_allow_html=True)
    if df is None:
        st.warning("Dataset not found.")
        return

    feature_names = metadata.get("feature_names", list(FEATURE_CONFIG.keys()))
    eval_data = get_evaluation_data(model, scaler, feature_names)

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.plotly_chart(create_disease_pie(df), use_container_width=True)
    with row1_c2:
        st.plotly_chart(create_feature_importance_chart(model, feature_names), use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.plotly_chart(create_age_histogram(df), use_container_width=True)
    with row2_c2:
        st.plotly_chart(create_age_chol_scatter(df), use_container_width=True)

    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

    if eval_data[0] is not None:
        y_test, y_pred, y_prob, _ = eval_data
        row3_c1, row3_c2 = st.columns(2)
        with row3_c1:
            st.plotly_chart(create_roc_curve(y_test, y_prob), use_container_width=True)
        with row3_c2:
            st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred), use_container_width=True)


def render_history_page():
    st.markdown('<div class="section-title">📋 Patient Prediction History</div>', unsafe_allow_html=True)
    history = st.session_state.prediction_history

    if not history:
        st.markdown(
            """
            <div class="glass-card">
                <h3>No predictions yet</h3>
                <p style="color:#94a3b8;">Go to the Prediction page to run your first risk assessment.
                All predictions will be stored in this session.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.session_state.last_prediction = None
        st.rerun()

    records = []
    for i, rec in enumerate(history):
        records.append({
            "#": i + 1,
            "Timestamp": rec["timestamp"],
            "Risk Level": rec["risk_label"],
            "Probability (%)": round(rec["disease_prob"] * 100, 2),
            "Confidence (%)": round(rec["confidence"], 2),
            "Age": rec["inputs"]["age"],
            "Sex": FEATURE_CONFIG["sex"]["options"][rec["inputs"]["sex"]],
        })
    hist_df = pd.DataFrame(records)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    if history:
        st.download_button(
            "📊 Export History as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )


def render_model_insights_page(model, metadata):
    st.markdown('<div class="section-title">🧠 Model Insights</div>', unsafe_allow_html=True)

    best_name = metadata.get("best_model_name", "Unknown")
    metrics = metadata.get("metrics", {})
    feature_names = metadata.get("feature_names", list(FEATURE_CONFIG.keys()))

    st.markdown(
        f"""
        <div class="glass-card">
            <h3>🏆 Best Model: {best_name}</h3>
            <p style="color:#94a3b8;">Selected based on highest ROC-AUC across Logistic Regression,
            Random Forest, SVM, and XGBoost.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_items = [
        ("Accuracy", metrics.get("Accuracy", 0)),
        ("Precision", metrics.get("Precision", 0)),
        ("Recall", metrics.get("Recall", 0)),
        ("F1 Score", metrics.get("F1 Score", 0)),
        ("ROC-AUC", metrics.get("ROC-AUC", 0)),
    ]
    colors = ["blue", "green", "orange", "purple", "teal"]
    for col, (name, val), color in zip([c1, c2, c3, c4, c5], metric_items, colors):
        fmt = f"{val:.1%}" if name != "ROC-AUC" else f"{val:.4f}"
        with col:
            st.markdown(kpi_card(name, fmt, color), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔝 Top Important Features</div>', unsafe_allow_html=True)
    imp_df = get_feature_importance(model, feature_names)
    top5 = imp_df.head(5)
    for _, row in top5.iterrows():
        pct = row["importance"] / imp_df["importance"].max() * 100
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display:flex; justify-content:space-between; color:#e2e8f0;">
                    <span><strong>{row['feature']}</strong></span>
                    <span>{row['importance']:.4f}</span>
                </div>
                {progress_bar_html(pct, '#818cf8')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(create_feature_importance_chart(model, feature_names), use_container_width=True)
    with col_b:
        fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance (Random Forest)", use_container_width=True)

    shap_path = os.path.join(MODELS_DIR, "shap_summary.png")
    st.markdown('<div class="section-title">🔬 SHAP Analysis</div>', unsafe_allow_html=True)
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="warning-card">
                SHAP summary plot not available. Run SHAP analysis during training to generate
                <code>models/shap_summary.png</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    results = metadata.get("results_df", {})
    if results:
        st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
        comparison_df = pd.DataFrame(results)
        st.dataframe(
            comparison_df.style.format("{:.4f}").background_gradient(cmap="viridis", subset=comparison_df.columns),
            use_container_width=True,
        )

    plot_files = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("ROC Curve", "roc_curve.png"),
        ("Precision-Recall Curve", "precision_recall_curve.png"),
    ]
    cols = st.columns(3)
    for i, (title, filename) in enumerate(plot_files):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with cols[i]:
                st.image(path, caption=title, use_container_width=True)


def render_dataset_page(df):
    st.markdown('<div class="section-title">📊 Dataset Analysis</div>', unsafe_allow_html=True)
    if df is None:
        st.warning("Dataset not found at dataset/heart.csv")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Rows", str(len(df)), "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Features", str(len(df.columns) - 1), "purple"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Classes", "2", "green"), unsafe_allow_html=True)
    with c4:
        missing = df.isnull().sum().sum()
        st.markdown(kpi_card("Missing", str(missing), "orange"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📈 Statistics", "📊 Charts"])
    with tab1:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>Target Distribution</h3>
                <p style="color:#94a3b8;">
                No Disease (0): { (df['target']==0).sum() } &nbsp;|&nbsp;
                Heart Disease (1): { (df['target']==1).sum() }
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab2:
        st.dataframe(df.describe().T, use_container_width=True)
        null_df = pd.DataFrame({"Column": df.columns, "Missing Values": df.isnull().sum().values})
        st.dataframe(null_df, use_container_width=True, hide_index=True)
    with tab3:
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.plotly_chart(create_disease_pie(df), use_container_width=True)
            st.plotly_chart(create_age_histogram(df), use_container_width=True)
        with chart_c2:
            st.plotly_chart(create_age_chol_scatter(df), use_container_width=True)
            corr_path = os.path.join(MODELS_DIR, "correlation_heatmap.png")
            if os.path.exists(corr_path):
                st.image(corr_path, caption="Correlation Heatmap", use_container_width=True)
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)


def render_about_page(metadata):
    st.markdown('<div class="section-title">ℹ About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h3>📌 Project Overview</h3>
            <p style="color:#94a3b8; line-height:1.8;">
            The Heart Disease Prediction System is an AI-powered healthcare analytics platform
            that assesses cardiovascular risk using machine learning. Built as a production-quality
            end-to-end ML pipeline — from data analysis to an interactive deployment dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🎯 Problem Statement</h3>
                <p style="color:#94a3b8; line-height:1.8;">
                Heart disease remains a leading cause of mortality worldwide. Early detection
                through data-driven risk assessment can support preventive care and clinical
                decision-making.
                </p>
            </div>
            <div class="glass-card">
                <h3>🎯 Objective</h3>
                <p style="color:#94a3b8; line-height:1.8;">
                Build an intelligent system that analyzes patient medical attributes and predicts
                heart disease risk with high accuracy, presented through a modern SaaS-style dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🛠 Technologies Used</h3>
                <p style="color:#94a3b8; line-height:2;">
                🐍 Python &nbsp;·&nbsp; 📊 Pandas &nbsp;·&nbsp; 🤖 Scikit-Learn<br>
                🌲 Random Forest &nbsp;·&nbsp; ⚡ XGBoost<br>
                🎨 Streamlit &nbsp;·&nbsp; 📈 Plotly<br>
                💾 Joblib &nbsp;·&nbsp; 📄 FPDF2
                </p>
            </div>
            <div class="glass-card">
                <h3>🔮 Future Scope</h3>
                <p style="color:#94a3b8; line-height:1.8;">
                • SHAP/LIME explainability &nbsp;·&nbsp; Deep learning models<br>
                • Multi-disease prediction &nbsp;·&nbsp; REST API deployment<br>
                • Cloud hosting (AWS/GCP) &nbsp;·&nbsp; User authentication<br>
                • Real-time EHR integration &nbsp;·&nbsp; Mobile app
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    best = metadata.get("best_model_name", "N/A")
    st.markdown(
        f"""
        <div class="success-card">
            ✅ Current best model: <strong>{best}</strong> —
            Accuracy {metadata.get('metrics', {}).get('Accuracy', 0):.1%}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">👤 Author Information</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <h3>Built for Portfolio & Professional Showcase</h3>
            <p style="color:#94a3b8; line-height:2;">
            🔗 <a href="https://github.com" style="color:#818cf8;">GitHub Repository</a><br>
            🔗 <a href="https://linkedin.com" style="color:#818cf8;">LinkedIn Profile</a>
            </p>
            <p style="color:#64748b; font-size:0.85rem; margin-top:1rem;">
            ⚠️ Disclaimer: This tool is for educational and demonstration purposes only.
            It is not a substitute for professional medical advice, diagnosis, or treatment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    inject_custom_css()

    model, scaler, metadata = load_artifacts()
    if model is None or scaler is None:
        st.error(
            "⚠️ Model not found! Train the model first:\n\n"
            "```bash\npython train_model.py\n```"
        )
        st.stop()

    metadata = metadata or {}
    df = load_dataset()

    render_sidebar(metadata)
    render_top_header(metadata)

    page = st.session_state.page
    if page == "dashboard":
        render_dashboard(metadata, df)
    elif page == "prediction":
        render_prediction_page(model, scaler, metadata)
    elif page == "analytics":
        render_analytics_page(model, scaler, metadata, df)
    elif page == "history":
        render_history_page()
    elif page == "insights":
        render_model_insights_page(model, metadata)
    elif page == "dataset":
        render_dataset_page(df)
    elif page == "about":
        render_about_page(metadata)


if __name__ == "__main__":
    main()
