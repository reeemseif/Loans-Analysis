import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# Optional Plotly
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False

# Small helper to pass safe kwargs to imshow
PASTEL_PALETTE = [
    "#AEC6CF", "#FFB7B2", "#FDFD96", "#B39EB5", "#77DD77", "#CFCFC4", "#FFD1DC", "#B5EAD7",
]

if _HAS_PLOTLY:
    _PX_KWARGS = {"template": "plotly_white", "color_discrete_sequence": PASTEL_PALETTE}
else:
    _PX_KWARGS = {"template": "plotly_white"}


def _px_kwargs_for(kind: str = "default"):
    if not _HAS_PLOTLY:
        return {}
    if kind == "imshow":
        return {k: v for k, v in _PX_KWARGS.items() if k in ("template", "color_continuous_scale")}
    return dict(_PX_KWARGS)


def _show_plotly_or_fallback(fig, fallback_df=None):
    if _HAS_PLOTLY and fig is not None:
        try:
            # ensure the default trace color sequence uses our pastel palette
            fig.update_layout(colorway=PASTEL_PALETTE)
        except Exception:
            # if fig doesn't accept update_layout in this env, ignore
            pass
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not available — showing table preview.")
        if fallback_df is not None:
            st.dataframe(fallback_df.head(100))


def render_metric(label: str, value: str):
    html = f"""
    <div style='line-height:1.1; margin-bottom:6px;'>
      <div style='font-size:13px; color:#6b6b6b;'>{label}</div>
      <div style='font-size:20px; font-weight:700; color:#111;'>{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(layout="wide")
st.title("Risk Analysis")

@st.cache_data
def load_data(path: str = "cleaned_df.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

# Load
DATA_PATH = "cleaned_df.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"{DATA_PATH} not found. Place the file in the project root.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

st.markdown("## Default rate analysis")

# Ensure loan_status exists
if "loan_status" in df.columns:
    charged_mask = df["loan_status"].astype(str).str.contains("Charged|Default|charged|default", case=False, na=False)
else:
    charged_mask = pd.Series(False, index=df.index)

# Default rates by grade

if "grade" in df.columns:
    by_grade = pd.concat([df["grade"], charged_mask.rename("is_default")], axis=1).dropna()
    grade_rates = by_grade.groupby("grade")["is_default"].mean().multiply(100).reset_index().rename(columns={"is_default":"default_pct"})
    grade_rates = grade_rates.sort_values("grade")
    # show percentages on bars
    if _HAS_PLOTLY:
        fig = px.bar(grade_rates, x="grade", y="default_pct", title="Default rate by grade (%)", labels={"default_pct":"% default"}, text="default_pct", **_PX_KWARGS)
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    else:
        fig = None
    _show_plotly_or_fallback(fig, fallback_df=grade_rates.assign(default_pct=grade_rates["default_pct"].round(2)))
else:
    st.write("No `grade` column available to compute default rates by grade.")

# Default rates by purpose
st.subheader("Default rate by loan purpose (top 10 riskiest)")
if "loan_purpose" in df.columns:
    purpose_df = pd.concat([df["loan_purpose"].fillna("(missing)"), charged_mask.rename("is_default")], axis=1)
    purpose_rates = purpose_df.groupby("loan_purpose")["is_default"].mean().multiply(100).reset_index().rename(columns={"is_default":"default_pct"})
    purpose_rates = purpose_rates.sort_values("default_pct", ascending=False).head(10)
    if _HAS_PLOTLY:
        fig2 = px.bar(purpose_rates, x="loan_purpose", y="default_pct", title="Top 10 riskiest loan purposes (by default %)", labels={"default_pct":"% default","loan_purpose":"Purpose"}, text="default_pct", **_PX_KWARGS)
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_xaxes(tickangle=45)
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    else:
        fig2 = None
    _show_plotly_or_fallback(fig2, fallback_df=purpose_rates.assign(default_pct=purpose_rates["default_pct"].round(2)))
else:
    st.write("No `loan_purpose` column available.")

# Default rates by homeownership
st.subheader("Default rate by homeownership")
if "homeownership" in df.columns:
    ho_df = pd.concat([df["homeownership"].fillna("(missing)"), charged_mask.rename("is_default")], axis=1)
    ho_rates = ho_df.groupby("homeownership")["is_default"].mean().multiply(100).reset_index().rename(columns={"is_default":"default_pct"})
    ho_rates = ho_rates.sort_values("default_pct", ascending=False)
    if _HAS_PLOTLY:
        fig3 = px.bar(ho_rates, x="homeownership", y="default_pct", title="Default rate by homeownership", labels={"default_pct":"% default","homeownership":"Homeownership"}, text="default_pct", **_PX_KWARGS)
        fig3.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    else:
        fig3 = None
    _show_plotly_or_fallback(fig3, fallback_df=ho_rates.assign(default_pct=ho_rates["default_pct"].round(2)))
else:
    st.write("No `homeownership` column available to compare default rates.")

st.markdown("---")
st.subheader("Tax liens and bankruptcy")
# Tax liens / bankruptcy pie or donut
pie_cols = {}
if "tax_liens" in df.columns:
    pie_cols["Tax liens (any)"] = (df["tax_liens"] > 0).mean()
if "public_record_bankrupt" in df.columns:
    pie_cols["Public bankrupt"] = (df["public_record_bankrupt"] > 0).mean()

if pie_cols:
    labels = list(pie_cols.keys())
    values = [pie_cols[k] for k in labels]
    pie_df = pd.DataFrame({"label": labels, "pct": [v * 100 for v in values]})
    if _HAS_PLOTLY:
        fig_p = px.pie(pie_df, names="label", values="pct", title="Share with tax liens / bankruptcies", hole=0.4, **_PX_KWARGS)
    else:
        fig_p = None
    _show_plotly_or_fallback(fig_p, fallback_df=pie_df)
else:
    st.write("No tax lien or bankruptcy columns available to summarize.")

st.markdown("---")
st.subheader("Risk factor correlation")
# Create derived features for correlation
corr_df = df.copy()
# utilization
if ("total_credit_limit" in corr_df.columns) and ("total_credit_utilized" in corr_df.columns):
    corr_df["credit_utilization_pct"] = (corr_df["total_credit_utilized"] / corr_df["total_credit_limit"]).replace([np.inf, -np.inf], np.nan) * 100
# select numeric risk-related columns
numeric_candidates = ["interest_rate","debt_to_income","delinq_2y","inquiries_last_12m","num_open_cc_accounts","credit_utilization_pct","loan_amount","installment"]
cols_present = [c for c in numeric_candidates if c in corr_df.columns]
if len(cols_present) >= 2:
    corr_mat = corr_df[cols_present].corr().round(2)
    if _HAS_PLOTLY:
        fig_corr = px.imshow(corr_mat, text_auto=True, aspect="auto", title="Risk factor correlation", **_px_kwargs_for("imshow"))
    else:
        fig_corr = None
    _show_plotly_or_fallback(fig_corr, fallback_df=corr_mat)
else:
    st.write("Not enough numeric risk-related columns to compute correlation matrix.")

st.markdown("---")
st.subheader("High-risk borrower profile")
# Define simple heuristic for high risk: charged OR high DTI OR high utilization OR delinquencies
high_risk_mask = charged_mask.copy()
if "debt_to_income" in df.columns:
    high_risk_mask = high_risk_mask | (df["debt_to_income"].fillna(0) > 40)
if ("total_credit_limit" in df.columns) and ("total_credit_utilized" in df.columns):
    util_pct = (df["total_credit_utilized"] / df["total_credit_limit"]).replace([np.inf, -np.inf], np.nan) * 100
    high_risk_mask = high_risk_mask | (util_pct.fillna(0) > 80)
if "delinq_2y" in df.columns:
    high_risk_mask = high_risk_mask | (df["delinq_2y"].fillna(0) > 0)

hr = df[high_risk_mask].copy()
if hr.empty:
    st.write("No high-risk borrowers identified by the simple heuristic.")
else:
    st.write(f"Found {len(hr)} high-risk borrowers (simple heuristic). Showing top 20 by interest rate.")
    if "interest_rate" in hr.columns:
        hr = hr.sort_values("interest_rate", ascending=False)
    display_cols = [c for c in ["emp_title","state","grade","loan_purpose","loan_amount","interest_rate","debt_to_income","delinq_2y"] if c in hr.columns]
    st.dataframe(hr[display_cols].head(20))
    # download
    csv = hr[display_cols].head(100).to_csv(index=True)
    st.download_button("Download high-risk sample (CSV)", data=csv, file_name="high_risk_sample.csv", mime="text/csv")

st.markdown("---")
st.markdown("Notes: the high-risk heuristic is intentionally simple. For production use, replace with a trained model or a more formal scoring rule.")
