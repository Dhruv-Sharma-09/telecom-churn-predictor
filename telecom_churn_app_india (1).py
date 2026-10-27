from pathlib import Path
from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background:#0b1020; color:#e6edf7; }
[data-testid="stHeader"] { background:rgba(11,16,32,.9); }
section[data-testid="stSidebar"] { background:#10182b; border-right:1px solid #25314a; }
.block-container { padding-top:1.4rem; max-width:1500px; }
h1,h2,h3 { letter-spacing:-.025em; }
.hero { padding:1.4rem 1.6rem; border:1px solid #293753; border-radius:18px;
 background:linear-gradient(115deg,#172443 0%,#111a2e 55%,#102d3c 100%); margin-bottom:1.2rem; }
.eyebrow { color:#55d6be; text-transform:uppercase; letter-spacing:.14em; font-size:.76rem; font-weight:700; }
.hero-title { font-size:2rem; font-weight:800; margin:.25rem 0; color:#f3f7ff; }
.hero-sub { color:#aab8cf; margin:0; }
.kpi { background:linear-gradient(145deg,#151f34,#111827); border:1px solid #2a3852;
 border-radius:16px; padding:1.05rem 1.15rem; min-height:125px; }
.kpi-label { color:#9eacc3; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
.kpi-value { color:#f4f7ff; font-size:2rem; font-weight:800; margin:.3rem 0; }
.kpi-note { color:#8fa2bf; font-size:.83rem; }
div[data-testid="stForm"] { border:1px solid #293753; border-radius:14px; padding:1rem; background:#10182b; }
.stButton>button,.stDownloadButton>button { border-radius:10px; border:1px solid #2b9d98; background:#167d79; color:white; font-weight:700; }
.stButton>button:hover,.stDownloadButton>button:hover { background:#1b958f; border-color:#55d6be; color:white; }
[data-testid="stDataFrame"] { border:1px solid #293753; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dce7f7"),
    margin=dict(l=20, r=20, t=75, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

TARGET_CANDIDATES = ["churn", "Churn", "CHURN", "customer_churn", "ChurnStatus", "churn_status"]
ID_CANDIDATES = ["customer_id", "customerID", "CustomerID", "customer_id"]

def load_csv(file_bytes):
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(
            "Could not identify the churn target column. Expected a column named 'churn' or 'Churn'. "
            "Please upload a dataset that contains the actual churn outcome."
        )
    df = df.dropna(subset=[target]).copy()
    # Normalize common binary churn labels to 0/1 while preserving target semantics.
    vals = set(df[target].astype(str).str.strip().str.lower().unique())
    if vals.issubset({"0", "1", "0.0", "1.0", "true", "false"}):
        df[target] = df[target].astype(str).str.strip().str.lower().map(
            {"0": 0, "0.0": 0, "false": 0, "1": 1, "1.0": 1, "true": 1}
        )
    elif vals.issubset({"yes", "no"}):
        df[target] = df[target].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    else:
        raise ValueError(
            f"Target column '{target}' must contain binary values (0/1 or Yes/No). "
            f"Found examples: {list(df[target].astype(str).unique()[:8])}"
        )
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)
    if df[target].nunique() != 2:
        raise ValueError("The churn target must contain both classes (0 = stayed, 1 = churned).")
    return df.reset_index(drop=True), target

def make_pipeline(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])

def train_model(data_bytes, df, target):
    # A content fingerprint avoids accidentally reusing a model after changing the uploaded CSV.
    fingerprint = hashlib.sha256(data_bytes).hexdigest()
    X = df.drop(columns=[target] + [c for c in ID_CANDIDATES if c in df.columns], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable input features found after excluding the churn target and customer ID.")
    y = df[target]
    if y.value_counts().min() < 2:
        raise ValueError("Each churn class needs at least two records to train and test the model.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(X)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob) if y_test.nunique() > 1 else float("nan"),
        "fingerprint": fingerprint,
    }
    return model, X.columns.tolist(), metrics

def metric_card(label, value, note, accent="#55d6be"):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{accent}">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("## 📡 Churn Intelligence")
st.sidebar.caption("CUSTOMER RETENTION ANALYTICS")
st.sidebar.divider()
default_candidates = [
    Path("/content/IndiaTelecomData.csv"),
    Path("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path(__file__).resolve().parent / "IndiaTelecomData.csv",
    Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
default_path = next((p for p in default_candidates if p.exists()), None)
uploaded_dataset = st.sidebar.file_uploader(
    "Upload churn dataset (CSV)", type=["csv"],
    help="Dataset must include a binary churn target column named churn or Churn."
)
if uploaded_dataset is not None:
    data_bytes = uploaded_dataset.getvalue()
    source_label = f"Uploaded: {uploaded_dataset.name}"
elif default_path is not None:
    data_bytes = default_path.read_bytes()
    source_label = f"Current file: {default_path.name}"
else:
    st.info("Upload a CSV dataset in the sidebar to get started.")
    st.stop()

if st.sidebar.button("↻ Refresh / reload dataset", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

try:
    df, target = load_csv(data_bytes)
    model, feature_cols, model_metrics = train_model(data_bytes, df, target)
except Exception as exc:
    st.error(f"Could not load/train model: {exc}")
    st.stop()

id_cols = [c for c in ID_CANDIDATES if c in df.columns]
feature_df = df.drop(columns=[target] + id_cols, errors="ignore")
churn_count = int(df[target].sum())
total_customers = len(df)
churn_rate = churn_count / total_customers if total_customers else 0

st.markdown("""
<div class="hero">
  <div class="eyebrow">Customer retention • Machine learning</div>
  <div class="hero-title">Telecom Churn Intelligence</div>
  <p class="hero-sub">Explore churn patterns, score customer risk, and identify retention opportunities.</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"Dataset: {source_label} · {total_customers:,} valid customers · Target column: {target}")

overview, predict_tab, batch_tab, analytics_tab = st.tabs(
    ["Overview", "Individual prediction", "Batch scoring", "Analytics"]
)

with overview:
    a, b, c, d = st.columns(4)
    with a: metric_card("Total customers", f"{total_customers:,}", "Valid records", "#dbeafe")
    with b: metric_card("Churned customers", f"{churn_count:,}", "Target label = 1", "#fb7185")
    with c: metric_card("Overall churn rate", f"{churn_rate:.2%}", "Historical dataset rate", "#fbbf24")
    with d: metric_card("Model ROC-AUC", f"{model_metrics['auc']:.3f}" if np.isfinite(model_metrics["auc"]) else "N/A", "Held-out test split", "#55d6be")

    left, right = st.columns(2)
    with left:
        mix = df[target].map({0: "Stayed", 1: "Churned"}).value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(mix, names="Status", values="Customers", hole=.65, title="Customer churn mix",
                     color="Status", color_discrete_map={"Stayed": "#3b82f6", "Churned": "#fb7185"})
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        possible_groups = [c for c in ["telecom_partner", "Contract", "InternetService", "state", "city", "gender"] if c in df.columns]
        if not possible_groups:
            possible_groups = [c for c in df.select_dtypes(exclude=["number"]).columns if c != target]
        if possible_groups:
            group_col = possible_groups[0]
            rates = df.groupby(group_col, dropna=False)[target].agg(["mean", "count"]).reset_index()
            rates = rates.rename(columns={"mean": "Churn rate", "count": "Customers"})
            rates["Churn rate label"] = rates["Churn rate"].map(lambda x: f"{x:.1%}")
            fig = px.bar(rates, x=group_col, y="Churn rate", color=group_col,
                         text="Churn rate label", title=f"Churn rate by {group_col.replace('_', ' ')}",
                         hover_data=["Customers"])
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat=".0%", showlegend=False)
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical feature available for a group comparison chart.")

    st.markdown("#### Model quality snapshot")
    m1, m2 = st.columns(2)
    m1.metric("Test accuracy", f"{model_metrics['accuracy']:.1%}")
    m2.metric("Test ROC-AUC", f"{model_metrics['auc']:.3f}" if np.isfinite(model_metrics["auc"]) else "N/A")
    st.caption("These metrics use a held-out test split and do not guarantee future performance.")

with predict_tab:
    st.subheader("Score an individual customer")
    st.caption("Enter values for the features available in the uploaded dataset. The model estimates churn risk; it does not guarantee an outcome.")
    with st.form("individual_prediction_form"):
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            s = feature_df[col]
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(s):
                    low = float(s.min())
                    high = float(s.max())
                    med = float(s.median())
                    if not np.isfinite(low) or not np.isfinite(high):
                        inputs[col] = med if np.isfinite(med) else 0.0
                        st.number_input(col.replace("_", " ").title(), value=float(inputs[col]), key=f"num_{col}")
                    elif low == high:
                        inputs[col] = st.number_input(col.replace("_", " ").title(), value=low, key=f"num_{col}")
                    else:
                        val = st.number_input(
                            col.replace("_", " ").title(),
                            min_value=low, max_value=high, value=min(max(med, low), high),
                            key=f"num_{col}"
                        )
                        inputs[col] = val
                else:
                    opts = s.dropna().astype(str).value_counts().index.tolist()
                    if not opts:
                        opts = ["Unknown"]
                    inputs[col] = st.selectbox(col.replace("_", " ").title(), opts[:100], key=f"cat_{col}")
        submitted = st.form_submit_button("Estimate churn risk", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{c: inputs.get(c) for c in feature_cols}])
        probability = float(model.predict_proba(sample)[0, 1])
        label = "Higher risk" if probability >= 0.5 else "Lower risk"
        st.markdown("### Prediction result")
        p1, p2 = st.columns(2)
        p1.metric("Estimated churn probability", f"{probability:.1%}")
        p2.metric("Risk category", label)
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Risk category uses a 50% probability threshold; choose an operational threshold based on validation and business needs.")

with batch_tab:
    st.subheader("Batch scoring")
    st.write("Upload a CSV with the same feature columns as the training dataset. The churn target and customer ID are optional.")
    batch_file = st.file_uploader("Upload customers to score", type=["csv"], key="batch_uploader")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_features = [c for c in feature_cols if c not in batch_df.columns]
            if missing_features:
                st.error("Batch file is missing feature columns: " + ", ".join(missing_features))
            else:
                scored = batch_df.copy()
                probabilities = model.predict_proba(batch_df[feature_cols])[:, 1]
                scored["predicted_churn_probability"] = probabilities
                scored["predicted_churn"] = (probabilities >= 0.5).astype(int)
                st.dataframe(scored.head(100), use_container_width=True)
                st.download_button(
                    "Download scored CSV", data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="telecom_churn_scored.csv", mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not score batch file: {exc}")

with analytics_tab:
    st.subheader("Explore the dataset")
    f1, f2 = st.columns(2)
    with f1:
        categorical_cols = [c for c in df.select_dtypes(exclude=["number"]).columns if c not in id_cols]
        chosen_cat = st.selectbox("Break down by category", ["(none)"] + categorical_cols)
    with f2:
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target and c not in id_cols]
        chosen_num = st.selectbox("Numeric feature distribution", ["(none)"] + numeric_cols)
    if chosen_cat != "(none)":
        summary = df.groupby(chosen_cat, dropna=False)[target].agg(["mean", "count"]).reset_index()
        summary.columns = [chosen_cat, "Churn rate", "Customers"]
        summary["Churn rate"] = summary["Churn rate"] * 100
        st.markdown(f"**Churn rate by {chosen_cat}**")
        st.dataframe(summary.sort_values("Churn rate", ascending=False), use_container_width=True)
        chart = px.bar(summary, x=chosen_cat, y="Churn rate", color="Churn rate",
                       title=f"Churn rate by {chosen_cat}", hover_data=["Customers"])
        chart.update_layout(**PLOTLY_LAYOUT, yaxis_title="Churn rate (%)")
        st.plotly_chart(chart, use_container_width=True)
    if chosen_num != "(none)":
        chart = px.histogram(df, x=chosen_num, color=target.map({0:"Stayed",1:"Churned"}),
                             barmode="overlay", opacity=.75, title=f"Distribution of {chosen_num}")
        chart.update_layout(**PLOTLY_LAYOUT, legend_title="Status")
        st.plotly_chart(chart, use_container_width=True)
    st.markdown("#### Data preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "Download current dataset", data=df.to_csv(index=False).encode("utf-8"),
        file_name="current_telecom_churn_dataset.csv", mime="text/csv"
    )
