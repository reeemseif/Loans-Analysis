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

# Safe defaults for Plotly kwargs
if _HAS_PLOTLY:
    try:
        PASTEL_PALETTE = [
            "#AEC6CF", "#FFB7B2", "#FDFD96", "#B39EB5", "#77DD77", "#CFCFC4", "#FFD1DC", "#B5EAD7",
        ]
        _PX_KWARGS = {"template": "plotly_white", "color_discrete_sequence": PASTEL_PALETTE}
    except Exception:
        _PX_KWARGS = {"template": "plotly_white"}
else:
    _PX_KWARGS = {}


def _px_kwargs_for(kind: str = "default"):
    if not _HAS_PLOTLY:
        return {}
    if kind == "imshow":
        return {k: v for k, v in _PX_KWARGS.items() if k in ("template", "color_continuous_scale")}
    return dict(_PX_KWARGS)


def _show_plotly_or_fallback(fig, fallback_df=None):
    if _HAS_PLOTLY and fig is not None:
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
st.title("Loan Performance")

@st.cache_data
def load_data(path: str = "cleaned_df.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

# Load dataset
DATA_PATH = "cleaned_df.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"{DATA_PATH} not found. Place the file in the project root.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# --- Sidebar filters for Loan Performance -------------------------------
st.sidebar.header("Loan Performance filters")
# sampling for heavy plots
sample_frac = st.sidebar.slider("Sample fraction for heavy plots (scatter)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)

# categorical filters (guarded by existence)
grade_opts = sorted(df["grade"].dropna().unique()) if "grade" in df.columns else []
grade_filter = st.sidebar.multiselect("Filter grades (leave empty for all)", options=grade_opts)

term_opts = sorted(df["term"].dropna().unique()) if "term" in df.columns else []
term_filter = st.sidebar.multiselect("Filter terms (leave empty for all)", options=term_opts)

purpose_opts = sorted(df["loan_purpose"].dropna().unique()) if "loan_purpose" in df.columns else []
purpose_filter = st.sidebar.multiselect("Filter loan purposes (leave empty for all)", options=purpose_opts)

# (issue_month sidebar filter removed per request)

# apply filters to a working copy
plot_df = df.copy()
if grade_filter:
    plot_df = plot_df[plot_df["grade"].isin(grade_filter)]
if term_filter:
    plot_df = plot_df[plot_df["term"].isin(term_filter)]
if purpose_filter:
    plot_df = plot_df[plot_df["loan_purpose"].isin(purpose_filter)]
# issue_month filtering removed — charts that use issue_month still compute their own date parsing locally

# Top-level KPIs (reflecting filters)
st.markdown("## Key performance metrics")
col1, col2, col3 = st.columns(3)
with col1:
    try:
        avg_ir = plot_df["interest_rate"].dropna().mean()
        render_metric("Avg interest rate", f"{avg_ir:.2f}%")
    except Exception:
        render_metric("Avg interest rate", "N/A")
with col2:
    try:
        med_loan = plot_df["loan_amount"].dropna().median()
        render_metric("Median loan amount", f"${med_loan:,.0f}")
    except Exception:
        render_metric("Median loan amount", "N/A")
with col3:
    try:
        med_inst = plot_df["installment"].dropna().median()
        render_metric("Median installment", f"${med_inst:,.0f}")
    except Exception:
        render_metric("Median installment", "N/A")

st.markdown("---")

# 1) Interest rate trends (by issue_month if present)
st.subheader("Interest rate trends over time")
if "issue_month" in df.columns:
    im = df["issue_month"].dropna().astype(str)
    # try to parse common formats like 'Feb-2018'
    try:
        im_dt = pd.to_datetime(im, format="%b-%Y", errors="coerce")
    except Exception:
        im_dt = pd.to_datetime(im, errors="coerce")
    df_tmp = df.copy()
    df_tmp["issue_month_dt"] = im_dt
    grp = df_tmp.dropna(subset=["issue_month_dt", "interest_rate"]).groupby(pd.Grouper(key="issue_month_dt", freq="M"))["interest_rate"].mean().reset_index()
    if not grp.empty:
        if _HAS_PLOTLY:
            fig_ir = px.line(grp, x="issue_month_dt", y="interest_rate", title="Average interest rate over time", labels={"issue_month_dt":"Issue month","interest_rate":"Avg interest rate (%)"}, **_PX_KWARGS)
            fig_ir.update_traces(mode="lines+markers")
        else:
            fig_ir = None
        _show_plotly_or_fallback(fig_ir, fallback_df=grp)
    else:
        st.write("Not enough issue_month + interest_rate data to show trend.")
else:
    st.write("No `issue_month` column available to plot interest rate trends.")

st.markdown("---")

# 2) Loan amount distributions
st.subheader("Loan amount distribution")
try:
    if _HAS_PLOTLY:
        fig_la = px.histogram(df, x="loan_amount", nbins=60, title="Loan amount distribution", **_PX_KWARGS)
    else:
        fig_la = None
    _show_plotly_or_fallback(fig_la, fallback_df=df[["loan_amount"]].dropna())
except Exception:
    st.write("Loan amount visualization not available.")

st.markdown("---")

# 3) Term comparison (36 vs 60 months)
st.subheader("Term comparison (36 vs 60 months)")
if "term" in df.columns:
    terms_of_interest = [36, 60]
    # coerce term to numeric if possible
    try:
        df["term_num"] = pd.to_numeric(df["term"], errors="coerce")
    except Exception:
        df["term_num"] = df["term"]
    comp = df[df["term_num"].isin(terms_of_interest)].dropna(subset=["term_num"])
    if not comp.empty:
        # median interest and median loan per term
        # use a list for column selection (pandas requires a list when selecting multiple columns)
        summary = comp.groupby("term_num")[["interest_rate", "loan_amount"]].median().reset_index().rename(columns={"interest_rate":"median_interest","loan_amount":"median_loan"})
        if _HAS_PLOTLY:
            fig_term1 = px.bar(summary, x="term_num", y="median_interest", title="Median interest rate by term (months)", labels={"term_num":"Term (months)","median_interest":"Median interest rate (%)"}, **_PX_KWARGS)
            fig_term2 = px.bar(summary, x="term_num", y="median_loan", title="Median loan amount by term (months)", labels={"term_num":"Term (months)","median_loan":"Median loan"}, **_PX_KWARGS)
        else:
            fig_term1 = fig_term2 = None
        _show_plotly_or_fallback(fig_term1, fallback_df=summary[["term_num","median_interest"]])
        _show_plotly_or_fallback(fig_term2, fallback_df=summary[["term_num","median_loan"]])
    else:
        st.write("No 36/60 term rows found in the data.")
else:
    st.write("No `term` column available to compare.")

st.markdown("---")

# 4) Installment analysis
st.subheader("Installment analysis")
if "installment" in df.columns:
    try:
        if _HAS_PLOTLY:
            fig_inst = px.histogram(df, x="installment", nbins=50, title="Installment distribution", **_PX_KWARGS)
        else:
            fig_inst = None
        _show_plotly_or_fallback(fig_inst, fallback_df=df[["installment"]].dropna())
        # payment burden by grade (median)
        if "grade" in df.columns:
            pb = df.dropna(subset=["installment","annual_income","grade"]).copy()
            pb["payment_burden_pct"] = (pb["installment"] / (pb["annual_income"]/12.0)).replace([np.inf,-np.inf], np.nan)*100
            med_pb = pb.groupby("grade")["payment_burden_pct"].median().reset_index().sort_values("grade")
            if _HAS_PLOTLY:
                fig_pb = px.bar(med_pb, x="grade", y="payment_burden_pct", title="Median payment burden by grade (%)", labels={"payment_burden_pct":"Median payment burden (%)","grade":"Grade"}, **_PX_KWARGS)
            else:
                fig_pb = None
            _show_plotly_or_fallback(fig_pb, fallback_df=med_pb)
    except Exception:
        st.write("Installment analysis failed to run.")
else:
    st.write("No `installment` column available.")

st.markdown("---")

# 5) Loan purpose breakdown
st.subheader("Loan purpose breakdown (top 20)")
if "loan_purpose" in df.columns:
    purpose_counts = df["loan_purpose"].fillna("(missing)").value_counts().nlargest(20).reset_index()
    purpose_counts.columns = ["loan_purpose","count"]
    if _HAS_PLOTLY:
        fig_pur = px.bar(purpose_counts, x="loan_purpose", y="count", title="Top loan purposes", labels={"loan_purpose":"Purpose","count":"Count"}, **_PX_KWARGS)
        fig_pur.update_xaxes(tickangle=45)
    else:
        fig_pur = None
    _show_plotly_or_fallback(fig_pur, fallback_df=purpose_counts)
else:
    st.write("No `loan_purpose` column available.")

st.markdown("---")

# 6) Grade distribution
st.subheader("Grade distribution")
if "grade" in df.columns:
    grade_counts = df["grade"].value_counts().sort_index().reset_index()
    grade_counts.columns = ["grade","count"]
    if _HAS_PLOTLY:
        fig_grade = px.bar(grade_counts, x="grade", y="count", title="Grade distribution", labels={"grade":"Grade","count":"Count"}, **_PX_KWARGS)
    else:
        fig_grade = None
    _show_plotly_or_fallback(fig_grade, fallback_df=grade_counts)
else:
    st.write("No `grade` column available.")

st.markdown("---")
st.markdown("Notes: These charts show high-level loan performance metrics. For deeper analysis you can add time windows, cohort segmentation, or model-backed risk scores.")
