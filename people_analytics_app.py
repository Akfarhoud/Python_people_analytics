"""people_analytics_app.py

People Analytics Suite  v3 – "Insights for Everyone" 🧑‍💼📊

A Streamlit app that turns raw HR spreadsheets into actionable insight cards for non‑experts:

Attrition Risk Radar – predicts who’s likely to leave and why (coefficients shown).

Engagement Health Check – scores survey data and flags good/bad pockets.

Auto‑Model Benchmark + SHAP – still there for power users.

One‑click CSV downloads for predictions & coefficients.



---

Run it ⬇️ 

pip install streamlit pandas numpy scikit-learn plotly shap xgboost lazypredict
streamlit run people_analytics_app.py

"""

from future import annotations

import io from typing import List import pandas as pd import numpy as np import streamlit as st import plotly.express as px from sklearn.model_selection import train_test_split, cross_val_score from sklearn.metrics import ( accuracy_score, roc_auc_score, mean_squared_error, r2_score, ) from sklearn.preprocessing import LabelEncoder, StandardScaler from sklearn.linear_model import LogisticRegression, LinearRegression from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor from sklearn.inspection import permutation_importance

Optional heavies

try: import shap  # type: ignore except ImportError: shap = None try: import xgboost as xgb  # type: ignore except ImportError: xgb = None

############################################################################### st.set_page_config(page_title="People Analytics Suite", page_icon="📊", layout="wide")

st.sidebar.title("📊 People Analytics Suite v3") file = st.sidebar.file_uploader("Upload people data (CSV)", type=["csv"])

###############################################################################

Helpers

############################################################################### @st.cache_data(show_spinner=False) def load_data(f): return pd.read_csv(f)

@st.cache_data(show_spinner=False) def encode_categoricals(df: pd.DataFrame): encoders = {} df_enc = df.copy() for col in df_enc.select_dtypes(exclude=np.number).columns: le = LabelEncoder() df_enc[col] = le.fit_transform(df_enc[col].astype(str)) encoders[col] = le return df_enc, encoders

############################################################################### if file is None: st.info("⬅️ Upload a CSV to begin.") st.stop()

with st.spinner("Loading data …"): df = load_data(file)

###############################################################################

Tabs

############################################################################### DESC, ATTRITION, ENGAGEMENT, ADVANCED = st.tabs([ "Descriptive", "Attrition Risk", "Engagement Health", "Advanced (Auto‑Model)"] )

###############################################################################

1️⃣ DESCRIPTIVE TAB

############################################################################### with DESC: st.header("Dataset Snapshot") col1, col2 = st.columns(2) with col1: st.metric("Rows", df.shape[0]) st.metric("Columns", df.shape[1]) st.write(df.head()) with col2: st.subheader("Missing values") st.write(df.isna().sum().to_frame("NA").T)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    sel = st.selectbox("Numeric column for histogram", num_cols, key="hist")
    st.plotly_chart(px.histogram(df, x=sel, nbins=30, title=f"{sel} – distribution"), use_container_width=True)

###############################################################################

2️⃣ ATTRITION RISK TAB

############################################################################### with ATTRITION: st.header("Attrition Risk Radar 🛑") st.write("Upload a dataset that contains a binary Attrition / Turnover flag (1 = left, 0 = stayed). If the column name differs, select it below.")

target_attr = st.selectbox("Attrition flag column", df.columns)
feature_cols_attr = [c for c in df.columns if c != target_attr]
cat_cols = df[feature_cols_attr].select_dtypes(exclude=np.number).columns.tolist()

if st.button("Train Attrition Model"):
    df_enc, encoders = encode_categoricals(df[[target_attr] + feature_cols_attr])
    X = df_enc[feature_cols_attr]
    y = df_enc[target_attr]

    # Split & scale (for coefficients)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic for coefficients & interpretability
    logreg = LogisticRegression(max_iter=2000, solver="lbfgs")
    logreg.fit(X_train_s, y_train)
    probs = logreg.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("ROC AUC", f"{auc:.3f}")

    # Coefficient table
    coefs = pd.Series(logreg.coef_[0], index=feature_cols_attr).sort_values(key=abs)
    coefs_df = coefs.rename("LogReg Coeff").to_frame()
    st.subheader("Feature impact (Logistic coefficients)")
    st.dataframe(coefs_df.style.background_gradient(cmap="RdBu", vmin=-abs(coefs).max(), vmax=abs(coefs).max(), axis=0))

    # Bar chart of top drivers
    top_n = st.slider("Show top N drivers", 5, min(20, len(coefs)), 10)
    st.plotly_chart(px.bar(coefs.tail(top_n), title="Strongest attrition drivers", orientation="h"), use_container_width=True)

    # Predict full data for user download & dashboard
    full_scaled = scaler.transform(X)
    df["attrition_risk"] = logreg.predict_proba(full_scaled)[:, 1]
    df["risk_flag"] = np.where(df.attrition_risk >= 0.5, "⚠️ High", "✅ Low")

    st.subheader("Overall picture")
    high_pct = (df.risk_flag == "⚠️ High").mean() * 100
    st.metric("High‑risk employees %", f"{high_pct:.1f}%")
    st.write("Employees flagged high‑risk (top 10):")
    st.dataframe(df.sort_values("attrition_risk", ascending=False).head(10))

    # Download
    st.download_button("Download risk predictions", df[["attrition_risk", "risk_flag"] + feature_cols_attr].to_csv(index=False).encode(), "attrition_risk.csv")

###############################################################################

3️⃣ ENGAGEMENT HEALTH TAB

############################################################################### with ENGAGEMENT: st.header("Engagement Health Check 💚") st.write("Select one or more engagement survey columns (Likert 1‑5 or %). We'll create a composite score and rate overall engagement.")

eng_cols = st.multiselect("Survey columns", df.columns)
if eng_cols:
    composite = df[eng_cols].mean(axis=1)
    df["engagement_score"] = composite
    # Simple thresholds
    df["engagement_flag"] = pd.cut(
        composite,
        bins=[-np.inf, 3, 4, np.inf],
        labels=["❌ Low", "⚠️ Medium", "✅ High"],
    )
    st.subheader("Overview")
    counts = df.engagement_flag.value_counts().reindex(["❌ Low", "⚠️ Medium", "✅ High"]).fillna(0)
    st.plotly_chart(px.pie(counts, values=counts.values, names=counts.index, title="Engagement levels"), use_container_width=True)
    st.metric("Avg score", f"{composite.mean():.2f}")
    st.metric("% High", f"{(df.engagement_flag=='✅ High').mean()*100:.1f}%")
    st.metric("% Low", f"{(df.engagement_flag=='❌ Low').mean()*100:.1f}%")

    st.write("Employees with low engagement (top 10):")
    st.dataframe(df[df.engagement_flag == "❌ Low"].head(10))

    st.download_button("Download engagement scores", df[["engagement_score", "engagement_flag"] + eng_cols].to_csv(index=False).encode(), "engagement_scores.csv")
else:
    st.info("Pick at least one engagement survey column to proceed.")

###############################################################################

4️⃣ ADVANCED TAB (Auto‑Model, SHAP, etc.)

############################################################################### with ADVANCED: st.header("Advanced Modelling & SHAP 🔬") st.write("For data scientists and power users.")

target = st.selectbox("Target column", df.columns, key="adv_target")
features = [c for c in df.columns if c != target]

if st.button("Run Benchmark", key="adv_btn"):
    df_enc, _ = encode_categoricals(df[[target] + features])
    X, y = df_enc[features], df_enc[target]
    problem = "classification" if y.nunique() <= 10 and y.dtype != float else "regression"
    st.write(f"Detected {problem} problem (unique target values = {y.nunique()}).")

    model = RandomForestClassifier(n_estimators=300) if problem == "classification" else RandomForestRegressor(n_estimators=300)
    model.fit(X, y)

    st.success("Model trained on full data.")
    result = permutation_importance(model, X, y, n_repeats=10, n_jobs=-1)
    imp_series = pd.Series(result.importances_mean, index=X.columns).sort_values()
    st.plotly_chart(px.bar(imp_series.tail(15), orientation="h", title="Perm. importance – top 15"), use_container_width=True)

    if shap:
        with st.spinner("Calculating SHAP (may take time)…"):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
        st.subheader("SHAP global importance")
        shap.plots.bar(shap_values, max_display=15, show=False)
        st.pyplot(shap.plots.bar(shap_values, max_display=15, show=False).figure)

###############################################################################

FOOTER

############################################################################### st.sidebar.markdown("---") st.sidebar.caption("© 2025 Abdulkarim Farhoud | v3 – attrition & engagement ready")

