"""app.py

People Analytics – Web Deploy Lite (v5)

A single-file Streamlit app designed for non-professionals to surface key People Analytics insights with minimal setup. Easy to deploy on Streamlit Cloud.

What’s new in v5

🧭 Column Mapper wizard (no fixed schema required)

📊 KPI Center: headcount trend, joiners, leavers, turnover

💸 Pay Equity Analyzer: unadjusted & adjusted gaps with controls

🧭 Performance x Engagement Quadrants (talent risk map)

🛌 Absence Dashboard

🔮 Attrition Prediction (coefficients & odds ratios, threshold slider)

💬 Plain-language Insight Cards + ⚙️ "What‑if" simulator


Minimal requirements

streamlit
pandas
numpy
scikit-learn
plotly

Run locally:

pip install -r requirements.txt
streamlit run app.py

""" from future import annotations

import numpy as np import pandas as pd import streamlit as st import plotly.express as px from typing import Tuple, Dict, Optional from datetime import datetime, date

from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler, LabelEncoder from sklearn.linear_model import LogisticRegression, LinearRegression from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix )

──────────────────────────────────────────────────────────────

PAGE & SIDEBAR

──────────────────────────────────────────────────────────────

st.set_page_config(page_title="People Analytics Lite", page_icon="📊", layout="wide") st.sidebar.title("People Analytics – Lite v5") st.sidebar.caption("Upload a CSV or use the built‑in demo data.")

──────────────────────────────────────────────────────────────

DEMO DATA (synthetic HR table)

──────────────────────────────────────────────────────────────

@st.cache_data def make_demo_data(n: int = 800) -> pd.DataFrame: rng = np.random.default_rng(42) dept = rng.choice(["Operations", "Sales", "Finance", "HR", "IT"], n) age = rng.integers(20, 60, n) hire = pd.to_datetime("2017-01-01") + pd.to_timedelta(rng.integers(0, 8*365, n), unit="D") tenure_yrs = (pd.to_datetime("today"):].class  # sentinel to avoid linter # Compute tenure based on hire and optional exit job_level = rng.integers(1, 6, n) overtime_hrs = np.round(rng.normal(6, 3, n).clip(0, 30), 1) training_hrs = np.round(rng.normal(20, 8, n).clip(0, 80), 1) salary = np.round(rng.normal(12000, 4000, n).clip(4000, 30000), -2)

# Engagement (Likert 1-5)
eng1 = rng.integers(1, 6, n)
eng2 = rng.integers(1, 6, n)
eng3 = rng.integers(1, 6, n)

# Simulate exits
logits = -2.0 + 0.07*overtime_hrs - 0.35*((eng1+eng2+eng3)/3 - 3) - 0.00003*salary + 0.03*(5-job_level)
prob = 1/(1+np.exp(-logits))
left_flag = (rng.uniform(0,1,n) < prob).astype(int)
# Build exit dates for those who left (after hire)
exit_dates = pd.to_datetime(np.where(left_flag==1,
    (hire + pd.to_timedelta(rng.integers(60, 6*365, n), unit="D")).astype("datetime64[ns]"),
    pd.NaT), utc=False)

df = pd.DataFrame({
    "EmployeeID": np.arange(1, n+1),
    "Dept": dept,
    "Age": age,
    "HireDate": hire,
    "ExitDate": exit_dates,
    "JobLevel": job_level,
    "OvertimeHours": overtime_hrs,
    "TrainingHours": training_hrs,
    "MonthlySalary": salary,
    "Engagement_Q1": eng1,
    "Engagement_Q2": eng2,
    "Engagement_Q3": eng3,
    "Attrition": left_flag,
    "Gender": rng.choice(["Female","Male"], n, p=[0.45,0.55])
})
# Tenure in years (if still employed, up to today)
today = pd.to_datetime("today").normalize()
df["TenureYears"] = ((df["ExitDate"].fillna(today) - df["HireDate"]) / np.timedelta64(1, 'Y')).round(2)
# Absence days/year synthetic
df["AbsenceDays"] = np.clip(np.round(rng.normal(6, 4, n)), 0, 40)
return df

──────────────────────────────────────────────────────────────

UTILITIES

──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False) def load_csv(f) -> pd.DataFrame: return pd.read_csv(f)

@st.cache_data(show_spinner=False) def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: df_enc = df.copy() enc: Dict[str, LabelEncoder] = {} for c in df_enc.select_dtypes(exclude=np.number).columns: le = LabelEncoder() df_enc[c] = le.fit_transform(df_enc[c].astype(str)) enc[c] = le return df_enc, enc

def to_datetime_safe(s: pd.Series) -> pd.Series: try: return pd.to_datetime(s, errors="coerce") except Exception: return pd.to_datetime(pd.Series([pd.NaT]*len(s)))

──────────────────────────────────────────────────────────────

DATA SOURCE

──────────────────────────────────────────────────────────────

source = st.sidebar.radio("Data source", ["Use demo data", "Upload CSV"], index=0) if source == "Upload CSV": file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if file is None: st.stop() df = load_csv(file) else: df = make_demo_data()

st.title("📊 People Analytics – Web Deploy Lite") st.caption("Descriptive • KPIs • Engagement • Attrition • Pay Equity • Insights")

──────────────────────────────────────────────────────────────

COLUMN MAPPER (Wizard)

──────────────────────────────────────────────────────────────

with st.expander("🔧 Column Mapper (set once)", expanded=True): cols = list(df.columns) def pick(label, hints): # try to guess column guess = next((c for c in cols if any(h in c.lower() for h in hints)), None) options = ["(None)"] + cols idx = 0 if guess is None else options.index(guess) return st.selectbox(label, options, index=idx)

emp_id_col   = pick("Employee ID", ["id", "employee", "emp_id"]) 
dept_col     = pick("Department", ["dept", "department", "function"]) 
gender_col   = pick("Gender", ["gender", "sex"]) 
hire_col     = pick("Hire date", ["hire", "join", "start"]) 
exit_col     = pick("Exit date", ["exit", "leave", "termination", "separation"]) 
salary_col   = pick("Salary (monthly or annual)", ["salary", "pay", "comp"])
perf_col     = pick("Performance rating", ["performance", "rating", "score"]) 
absence_col  = pick("Absence days (year)", ["absence", "sick", "leave_days"]) 
overtime_col = pick("Overtime hours", ["overtime"]) 
training_col = pick("Training hours", ["training"]) 
engagement_guess = [c for c in cols if any(k in c.lower() for k in ["engage", "enps", "q1", "q2", "q3"]) ]
eng_multi = st.multiselect("Engagement items (Likert 1–5 or %)", cols, default=engagement_guess)
attrition_flag_col = pick("Attrition flag (1 left / 0 stayed)", ["attrition", "turnover", "left", "leaver", "quit"])

Precompute helpful fields

if hire_col != "(None)": df[hire_col] = to_datetime_safe(df[hire_col]) if exit_col != "(None)": df[exit_col] = to_datetime_safe(df[exit_col])

Tenure (years)

if "TenureYears" not in df.columns: if hire_col != "(None)": today = pd.to_datetime("today").normalize() df["TenureYears"] = ((df[exit_col].fillna(today) - df[hire_col]) / np.timedelta64(1, 'Y')).round(2) if exit_col != "(None)" else ((today - df[hire_col]) / np.timedelta64(1, 'Y')).round(2)

──────────────────────────────────────────────────────────────

TABS

──────────────────────────────────────────────────────────────

TAB1, TAB2, TAB3, TAB4, TAB5, TAB6, TAB7 = st.tabs([ "Overview", "KPI Center", "Engagement", "Attrition", "Pay Equity", "What‑if", "Insights" ])

──────────────────────────────────────────────────────────────

OVERVIEW

──────────────────────────────────────────────────────────────

with TAB1: st.subheader("Dataset overview") c1, c2, c3 = st.columns(3) c1.metric("Rows", df.shape[0]) c2.metric("Columns", df.shape[1]) c3.metric("Missing cells", int(df.isna().sum().sum())) st.dataframe(df.head(), use_container_width=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    sel = st.selectbox("Numeric column for histogram", num_cols)
    st.plotly_chart(px.histogram(df, x=sel, nbins=30, title=f"Distribution of {sel}"), use_container_width=True)

──────────────────────────────────────────────────────────────

KPI CENTER – headcount, joiners, leavers, turnover

──────────────────────────────────────────────────────────────

with TAB2: st.subheader("KPI Center") if hire_col == "(None)": st.info("Set Hire date in Column Mapper to enable time-based KPIs.") else: min_d = pd.to_datetime(df[hire_col].min()).date() max_d = pd.to_datetime(df[exit_col].max()).date() if exit_col != "(None)" and df[exit_col].notna().any() else date.today() start, end = st.date_input("Date range", value=(min_d, max_d)) if isinstance(start, tuple): start, end = start rng = pd.date_range(start=start, end=end, freq="M")

# Monthly snapshots
    def monthly_headcount(snapshot: pd.Timestamp) -> int:
        if exit_col == "(None)":
            return int((df[hire_col] <= snapshot).sum())
        active = (df[hire_col] <= snapshot) & (df[exit_col].isna() | (df[exit_col] > snapshot))
        return int(active.sum())

    hc_series = pd.Series([monthly_headcount(m) for m in rng], index=rng)
    st.plotly_chart(px.line(hc_series, labels={"index":"Month","value":"Headcount"}, title="Headcount over time"), use_container_width=True)

    # Joiners & leavers per month
    joiners = df.groupby(pd.Grouper(key=hire_col, freq="M")).size().reindex(rng, fill_value=0)
    if exit_col != "(None)":
        leavers = df[df[exit_col].notna()].groupby(pd.Grouper(key=exit_col, freq="M")).size().reindex(rng, fill_value=0)
    else:
        leavers = pd.Series(0, index=rng)
    jl = pd.DataFrame({"Joiners": joiners, "Leavers": leavers})
    st.plotly_chart(px.bar(jl, barmode="group", title="Joiners vs Leavers"), use_container_width=True)

    avg_hc = hc_series.mean() if len(hc_series) else np.nan
    total_leavers = int(leavers.sum())
    turnover = (total_leavers / avg_hc * 100) if avg_hc and avg_hc > 0 else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg headcount", f"{avg_hc:.0f}" if not np.isnan(avg_hc) else "–")
    k2.metric("Leavers in range", f"{total_leavers}")
    k3.metric("Turnover % (period)", f"{turnover:.1f}%" if turnover==turnover else "–")

──────────────────────────────────────────────────────────────

ENGAGEMENT

──────────────────────────────────────────────────────────────

with TAB3: st.subheader("Engagement health") eng_cols = eng_multi if eng_cols: eng = df[eng_cols].apply(pd.to_numeric, errors="coerce") score = eng.mean(axis=1) df["EngagementScore"] = score # Normalize to 0–100 if score.max() <= 5: score_100 = (score - 1) / 4 * 100 else: score_100 = score.clip(0, 100) df["EngagementPct"] = np.round(score_100, 1) df["EngagementFlag"] = pd.cut(df["EngagementPct"], bins=[-1, 60, 80, 101], labels=["❌ Low", "⚠️ Medium", "✅ High"])

c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg engagement", f"{df['EngagementPct'].mean():.1f}%")
    c2.metric("High (%)", f"{(df['EngagementFlag']=='✅ High').mean()*100:.1f}%")
    c3.metric("Low (%)", f"{(df['EngagementFlag']=='❌ Low').mean()*100:.1f}%")
    c4.metric("Items used", len(eng_cols))

    st.plotly_chart(px.histogram(df, x="EngagementPct", nbins=30, title="Engagement score distribution"), use_container_width=True)

    # Performance x Engagement (talent risk map)
    if perf_col != "(None)":
        perf = pd.to_numeric(df[perf_col], errors="coerce")
        e = df["EngagementPct"]
        perf_thr = st.slider("Performance threshold (hi/lo)", 1.0, float(np.nanmax(perf.fillna(0))), 4.0, 0.1)
        eng_thr = st.slider("Engagement threshold (hi/lo %)", 0.0, 100.0, 75.0, 1.0)
        quadrant = np.select([
            (perf>=perf_thr) & (e>=eng_thr),
            (perf>=perf_thr) & (e<eng_thr),
            (perf<perf_thr) & (e>=eng_thr),
            (perf<perf_thr) & (e<eng_thr),
        ], ["⭐ Stars","⚠️ Flight-risk High Perf","🌱 Core Stable","❗ Underperforming"], default="Unclassified")
        st.plotly_chart(px.scatter(x=perf, y=e, color=quadrant, labels={"x":"Performance","y":"Engagement %"}, title="Performance × Engagement"), use_container_width=True)
        counts = pd.Series(quadrant).value_counts()
        st.write("Quadrant counts:")
        st.write(counts)
else:
    st.info("Select engagement items in the Column Mapper.")

──────────────────────────────────────────────────────────────

ATTRITION

──────────────────────────────────────────────────────────────

with TAB4: st.subheader("Attrition prediction (who might leave)") if attrition_flag_col == "(None)": st.info("Select Attrition flag in Column Mapper (1=left, 0=stayed). If you don't have a historical flag, you can still predict risk using current patterns, but accuracy will be limited.") # Features = all except target feature_cols = [c for c in df.columns if c != attrition_flag_col] # Encode categoricals for modelling X = df[feature_cols].copy() for c in X.select_dtypes(exclude=np.number).columns: X[c] = X[c].astype("category").cat.codes

# Prepare labels (binary)
if attrition_flag_col != "(None)":
    y_raw = df[attrition_flag_col]
    if set(pd.Series(y_raw).dropna().unique()).issubset({0,1}):
        y = y_raw
    else:
        pos_words = {"yes","y","left","leaver","quit","true","1","resigned","terminated"}
        y = y_raw.astype(str).str.lower().str.strip().map(lambda v: 1 if v in pos_words else 0)
else:
    # No label – use zeros as placeholder to enable scoring with caveat
    y = pd.Series(np.zeros(len(df), dtype=int))

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic (fast, explainable)
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_s, y_train)

# Metrics (if we have true labels)
if attrition_flag_col != "(None)":
    proba_test = logreg.predict_proba(X_test_s)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, preds_test):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, preds_test, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, preds_test, zero_division=0):.3f}")
    m4.metric("F1", f"{f1_score(y_test, preds_test, zero_division=0):.3f}")
    m5.metric("ROC AUC", f"{roc_auc_score(y_test, proba_test):.3f}")
    st.write("Confusion matrix @ 0.5:")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, preds_test), index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# Coefficients & Odds ratios
coefs = pd.Series(logreg.coef_[0], index=X.columns)
odds = np.exp(coefs)
coef_df = pd.DataFrame({"Coefficient": coefs, "OddsRatio (e^coef)": odds}).sort_values("Coefficient", key=lambda s: s.abs())
topN = st.slider("Show top N drivers", min_value=5, max_value=min(20, len(coef_df)), value=min(10, len(coef_df)))
st.dataframe(coef_df.tail(topN))
st.plotly_chart(px.bar(coef_df.tail(topN), y=coef_df.tail(topN).index, x="Coefficient", orientation="h", title="Logistic coefficients (direction & strength)"), use_container_width=True)

# Risk predictions for everyone
st.subheader("Predict risk for everyone")
threshold = st.slider("Risk threshold (High if ≥ threshold)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
proba_full = logreg.predict_proba(scaler.transform(X))[:, 1]
df["AttritionRisk"] = np.round(proba_full, 3)
df["RiskFlag"] = np.where(df["AttritionRisk"] >= threshold, "⚠️ High", "✅ Low")
st.metric("High-risk share", f"{(df['RiskFlag']=='⚠️ High').mean()*100:.1f}%")
st.plotly_chart(px.histogram(df, x="AttritionRisk", nbins=30, title="Predicted attrition risk (all employees)"), use_container_width=True)

st.download_button("Download risk predictions (CSV)", df[[attrition_flag_col] + ["AttritionRisk", "RiskFlag"] + feature_cols].to_csv(index=False).encode(), "attrition_risk.csv")
st.download_button("Download coefficients & odds ratios", coef_df.to_csv().encode(), "attrition_coefficients.csv")

# Save to session for What‑if
st.session_state["_model"] = logreg
st.session_state["_scaler"] = scaler
st.session_state["_Xcols"] = X.columns.tolist()

──────────────────────────────────────────────────────────────

PAY EQUITY ANALYZER – unadjusted & adjusted gaps

──────────────────────────────────────────────────────────────

with TAB5: st.subheader("Pay Equity Analyzer") if salary_col == "(None)": st.info("Select Salary in Column Mapper.") else: group_col = st.selectbox("Group by attribute", [c for c in [gender_col, dept_col] if c != "(None)"] + [c for c in df.select_dtypes(exclude=np.number).columns if c not in ["(None)", gender_col, dept_col]], index=0) if group_col: ref_group = st.selectbox("Reference group (baseline)", sorted(df[group_col].dropna().unique().tolist())) # Unadjusted gap (% difference in means vs ref) means = df.groupby(group_col)[salary_col].mean().sort_values() unadj = ((means / means.loc[ref_group] - 1) * 100).rename("Unadjusted % vs ref") st.write("Unadjusted pay vs reference:") st.dataframe(unadj.to_frame().round(2)) st.plotly_chart(px.bar(unadj, title="Unadjusted pay gap (%)"), use_container_width=True)

# Adjusted gap using linear regression on log(salary)
        work = df[[salary_col, group_col]].copy().dropna()
        work = work.join(df[["TenureYears", job] for job in []], how='left')  # no-op; placeholder
        X = pd.get_dummies(df[[group_col, dept_col] if dept_col != "(None)" else [group_col]], drop_first=False).fillna(0)
        # Set reference group by dropping its column set
        drop_cols = [c for c in X.columns if c.startswith(f"{group_col}_") and c != f"{group_col}_{ref_group}"]
        # Keep all and subtract baseline via reparameterization: we'll include all dummies and interpret as relative to ref via difference
        # Simpler: drop one level per categorical
        X = pd.get_dummies(df[[group_col] + ([dept_col] if dept_col != "(None)" else [])], drop_first=True)
        if "TenureYears" in df.columns:
            X = X.join(df[["TenureYears"]])
        if job_level := ("JobLevel" if "JobLevel" in df.columns else None):
            X = X.join(pd.get_dummies(df[[job_level]], drop_first=True))
        y = np.log(df[salary_col]).replace([np.inf, -np.inf], np.nan)
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        X2, y2 = X[mask], y[mask]
        lin = LinearRegression()
        lin.fit(X2, y2)
        coefs = pd.Series(lin.coef_, index=X2.columns)
        # Extract group effect columns
        grp_cols = [c for c in coefs.index if c.startswith(f"{group_col}_")]
        adj_pct = (np.exp(coefs[grp_cols]) - 1) * 100
        st.write("Adjusted gap (controls: Tenure, Dept, JobLevel if available). Values show % vs **reference level omitted by dummy coding**.")
        st.dataframe(adj_pct.sort_values().to_frame("Adjusted % vs reference").round(2))

──────────────────────────────────────────────────────────────

WHAT‑IF SIMULATOR

──────────────────────────────────────────────────────────────

with TAB6: st.subheader("What‑if simulator (attrition risk)") if "_model" not in st.session_state: st.info("Train the attrition model first (TAB: Attrition).") else: model = st.session_state["_model"] scaler = st.session_state["_scaler"] colsX = st.session_state["_Xcols"] # Build X matrix again X_all = df[colsX].copy() for c in X_all.select_dtypes(exclude=np.number).columns: X_all[c] = X_all[c].astype("category").cat.codes Xs = scaler.transform(X_all)

idx = st.number_input("Row index to simulate", 0, int(len(df)-1), 0)
    base_prob = float(model.predict_proba(Xs[[idx]])[0,1])
    st.metric("Current predicted risk", f"{base_prob:.3f}")

    # Choose up to 2 features to tweak
    tweak_feats = st.multiselect("Pick features to tweak", colsX, max_selections=2)
    if tweak_feats:
        vec = X_all.iloc[idx].copy()
        for feat in tweak_feats:
            fmin, fmax = float(df[feat].min()), float(df[feat].max())
            new_val = st.slider(f"Set {feat}", fmin, fmax, float(vec[feat]))
            vec[feat] = new_val
        X_new = scaler.transform(pd.DataFrame([vec], columns=colsX))
        new_prob = float(model.predict_proba(X_new)[0,1])
        delta = new_prob - base_prob
        if delta < 0:
            st.success(f"Risk decreases by {abs(delta):.3f} → **{new_prob:.3f}**")
        elif delta > 0:
            st.error(f"Risk increases by {delta:.3f} → **{new_prob:.3f}**")
        else:
            st.info("No change in risk.")

──────────────────────────────────────────────────────────────

INSIGHTS – auto-detected patterns & alerts

──────────────────────────────────────────────────────────────

with TAB7: st.subheader("Plain-language insights & alerts") # Engagement verdict if "EngagementPct" in df.columns: avg_e = df["EngagementPct"].mean() low_e = (df["EngagementFlag"] == "❌ Low").mean() * 100 if avg_e >= 80 and low_e < 10: st.success("Engagement looks good overall (avg ≥ 80% and <10% in Low).") elif avg_e < 60 or low_e >= 20: st.error("Engagement is concerning (avg < 60% or ≥20% in Low). Prioritise pulse checks and manager coaching.") else: st.warning("Engagement is mixed. Some teams need attention. Target Low group first.")

# Attrition verdict
if "AttritionRisk" in df.columns:
    high_share = (df["RiskFlag"] == "⚠️ High").mean() * 100
    if high_share >= 25:
        st.error("Attrition risk is **elevated** (≥25% flagged high). Prepare retention actions now.")
    elif high_share <= 10:
        st.success("Attrition risk is **under control** (≤10% flagged high). Keep monitoring.")
    else:
        st.warning("Attrition risk is **moderate**. Focus on top risk drivers and high-impact roles.")

# Department hotspots (leaver rate vs overall)
if dept_col != "(None)":
    overall_rate = (df[attrition_flag_col].mean()*100) if attrition_flag_col != "(None)" and df[attrition_flag_col].dropna().size else np.nan
    if overall_rate==overall_rate:  # not NaN
        grp = df.groupby(dept_col)[attrition_flag_col].mean().sort_values()*100
        hotspots = grp[grp >= overall_rate + 5]
        if len(hotspots):
            st.error(f"Hotspots: {', '.join([f'{k} ({v:.1f}%)' for k,v in hotspots.items()])} show **higher leaver rates** than overall ({overall_rate:.1f}%).")

# Pay gap alert
if salary_col != "(None)" and gender_col != "(None)":
    means = df.groupby(gender_col)[salary_col].mean()
    if len(means)>=2:
        ref = means.idxmax()  # use higher-paid group as ref
        gap = (means.min()/means.max()-1)*100
        if gap < -5:
            st.error(f"Pay gap alert: lower-paid gender group is **{abs(gap):.1f}%** below {ref}. Review pay equity.")

# Overtime ↔ risk correlation
if overtime_col != "(None)" and "AttritionRisk" in df.columns:
    corr = pd.to_numeric(df[overtime_col], errors="coerce").corr(df["AttritionRisk"]) 
    if pd.notna(corr) and corr > 0.2:
        st.warning(f"Overtime is positively correlated with attrition risk (r={corr:.2f}). Consider workload balancing.")

# Absence alert
if absence_col != "(None)":
    avg_abs = pd.to_numeric(df[absence_col], errors="coerce").mean()
    if pd.notna(avg_abs) and avg_abs > 10:
        st.warning(f"Absence is high on average (**{avg_abs:.1f} days/year**). Investigate wellbeing & attendance policies.")

st.sidebar.markdown("---") st.sidebar.caption("© 2025 – People Analytics Lite v5 | Built for easy cloud deployment")

