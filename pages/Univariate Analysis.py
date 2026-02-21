import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False

from typing import Optional

st.set_page_config(layout="wide")

st.title("Univariate — Cleaned Loans Dataset")


def render_metric(label: str, value: str, small_label: bool = False):
    """Render a consistent metric with HTML so font sizes are uniform.

    label: short label shown above the value (smaller text)
    value: main metric value (larger, bold)
    """
    html = f"""
    <div style='line-height:1.1; margin-bottom:6px;'>
        <div style='font-size:13px; color:#6b6b6b;'>{label}</div>
        <div style='font-size:20px; font-weight:700; color:#111;'>{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_question(n: int, question: str, answer: Optional[str] = None):
    """Render a prominent question and optional answer."""
    st.info(f"Question {n}: {question}")
    if answer is not None:
        st.markdown(f"**Answer:** {answer}")


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path, index_col=0)

# pastel palette used across pages for consistent, readable colors
PASTEL_PALETTE = ["#AEC6CF", "#FFB7B2", "#FDFD96", "#B39EB5", "#77DD77", "#CFCFC4", "#FFD1DC", "#B5EAD7"]
_PX_KWARGS = {"template": "plotly_white", "color_discrete_sequence": PASTEL_PALETTE}

DATA_PATH = "cleaned_df.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"{DATA_PATH} not found in the app folder. Place the file alongside `Home.py`.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# -- KPIs -------------------------------------------------
st.markdown("## Key dataset KPIs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric("Rows", f"{df.shape[0]:,}")
with col2:
    render_metric("Columns", f"{df.shape[1]}")
with col3:
    avg_income = df["annual_income"].dropna().mean()
    try:
        render_metric("Avg annual income", f"${avg_income:,.0f}")
    except Exception:
        render_metric("Avg annual income", "N/A")
with col4:
    med_loan = df["loan_amount"].dropna().median()
    try:
        render_metric("Median loan amount", f"${med_loan:,.0f}")
    except Exception:
        render_metric("Median loan amount", "N/A")

col5, col6, col7 = st.columns(3)
with col5:
    avg_ir = df["interest_rate"].dropna().mean()
    try:
        render_metric("Avg interest rate", f"{avg_ir:.2f}%")
    except Exception:
        render_metric("Avg interest rate", "N/A")
with col6:
    pct_tax_lien = (df["tax_liens"] > 0).mean() * 100
    render_metric("% with tax lien", f"{pct_tax_lien:.2f}%")
with col7:
    pct_bankrupt = (df["public_record_bankrupt"] > 0).mean() * 100
    render_metric("% with bankruptcies", f"{pct_bankrupt:.2f}%")

st.markdown("---")

# Sidebar controls
st.sidebar.header("Univariate controls")
num_bins = st.sidebar.slider("Histogram bins", min_value=10, max_value=200, value=50, step=10)
show_log = st.sidebar.checkbox("Log-scale for income histogram", value=False)

# -- Univariate plots -------------------------------------
st.markdown("## Numeric distributions")

# helper to show plotly or dataframe fallback
def _show_plotly_or_fallback(fig, fallback_df=None):
    if _HAS_PLOTLY and fig is not None:
        try:
            fig.update_layout(colorway=PASTEL_PALETTE)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Plot not available (Plotly not installed). Showing data preview instead.")
        if fallback_df is not None:
            st.dataframe(fallback_df.head(50))


# Annual income histogram
with st.container():
    st.subheader("Annual income")
    fig_income = px.histogram(df, x="annual_income", nbins=num_bins, title="Annual income distribution", **_PX_KWARGS)
    if show_log:
        fig_income.update_xaxes(type="log")
    st.plotly_chart(fig_income, use_container_width=True)

# Loan amount
with st.container():
    st.subheader("Loan amount")
    fig_loan = px.histogram(df, x="loan_amount", nbins=num_bins, title="Loan amount distribution", **_PX_KWARGS)
    st.plotly_chart(fig_loan, use_container_width=True)

# Interest rate histogram and simple stats (no boxplot)
with st.container():
    st.subheader("Interest rate")
    fig_ir_hist = None
    if _HAS_PLOTLY:
        fig_ir_hist = px.histogram(df, x="interest_rate", nbins=40, title="Interest rate distribution", **_PX_KWARGS)
    _show_plotly_or_fallback(fig_ir_hist, df[["interest_rate"]])
    # Show mean and median for readability
    try:
        mean_ir = df["interest_rate"].mean()
        med_ir = df["interest_rate"].median()
        st.markdown(f"- Mean interest rate: **{mean_ir:.2f}%**, Median interest rate: **{med_ir:.2f}%**")
    except Exception:
        st.write("Interest rate stats not available.")

st.markdown("---")
st.markdown("## Categorical summaries")

# loan_purpose counts
with st.container():
    st.subheader("Loan purposes (top 20)")
    if "loan_purpose" in df.columns:
        purpose_counts = df["loan_purpose"].value_counts().nlargest(20)
        fig_purpose = None
        if _HAS_PLOTLY:
            fig_purpose = px.bar(x=purpose_counts.index, y=purpose_counts.values, labels={"x":"Loan Purpose","y":"Count"}, title="Top loan purposes", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_purpose, purpose_counts.reset_index().rename(columns={"index":"purpose", "loan_purpose":"count"}))
    else:
        st.write("No loan_purpose column available.")

# homeownership
with st.container():
    st.subheader("Homeownership distribution")
    if "homeownership" in df.columns:
        ho_counts = df["homeownership"].value_counts()
        fig_ho = None
        if _HAS_PLOTLY:
            fig_ho = px.pie(values=ho_counts.values, names=ho_counts.index, title="Homeownership", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_ho, ho_counts.reset_index().rename(columns={"index":"homeownership", "homeownership":"count"}))
    else:
        st.write("No homeownership column available.")

# grade distribution
with st.container():
    st.subheader("Loan grade distribution")
    if "grade" in df.columns:
        grade_counts = df["grade"].value_counts().sort_index()
        fig_grade = None
        if _HAS_PLOTLY:
            fig_grade = px.bar(x=grade_counts.index, y=grade_counts.values, title="Grade counts", labels={"x":"Grade","y":"Count"}, **_PX_KWARGS)
        _show_plotly_or_fallback(fig_grade, grade_counts.reset_index().rename(columns={"index":"grade", "grade":"count"}))
    else:
        st.write("No grade column available.")

# experience years
with st.container():
    st.subheader("Experience (years)")
    if "experience_years" in df.columns:
        fig_exp = None
        if _HAS_PLOTLY:
            fig_exp = px.histogram(df, x="experience_years", nbins=40, title="Distribution of experience years", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_exp, df[["experience_years"]])
    else:
        st.write("No experience_years column available.")

# Top 10 states by count
with st.container():
    st.subheader("Top 10 states")
    if "state" in df.columns:
        state_counts = df["state"].value_counts().nlargest(10)
        fig_state = None
        if _HAS_PLOTLY:
            fig_state = px.bar(x=state_counts.index, y=state_counts.values, title="Top 10 states by borrower count", labels={"x": "State", "y": "Borrower Count"}, **_PX_KWARGS)
        _show_plotly_or_fallback(fig_state, state_counts.reset_index().rename(columns={"index":"state", "state":"count"}))
    else:
        st.write("No state column available.")

st.markdown("---")
st.markdown("## Additional univariate analyses & questions")

# Debt-to-income
with st.container():
    st.subheader("Debt-to-Income (DTI)")
    if "debt_to_income" in df.columns:
        fig_dti = None
        if _HAS_PLOTLY:
            fig_dti = px.histogram(df, x="debt_to_income", nbins=40, title="Debt-to-Income (DTI) distribution", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_dti, df[["debt_to_income"]])
        # Provide simple statistics instead of a boxplot for clarity
        try:
            mean_dti = df["debt_to_income"].mean()
            med_dti = df["debt_to_income"].median()
            st.markdown(f"- Mean DTI: **{mean_dti:.2f}%**, Median DTI: **{med_dti:.2f}%**")
        except Exception:
            st.write("DTI stats not available.")
        pct_high_dti = (df["debt_to_income"] > 30).mean() * 100
        show_question(1, "What proportion of borrowers have DTI > 30%?", f"**{pct_high_dti:.2f}%**")
    else:
        st.write("No debt_to_income column available.")

# Credit utilization percent (where available)
with st.container():
    st.subheader("Credit utilization")
    # compute percent utilization where limits are positive
    util = None
    if "total_credit_limit" in df.columns and (df["total_credit_limit"] > 0).any():
        util = (df["total_credit_utilized"] / df["total_credit_limit"]) * 100
        util = util.replace([np.inf, -np.inf], np.nan)
        fig_util = None
        if _HAS_PLOTLY:
            fig_util = px.histogram(util, nbins=40, labels={"value":"Utilization %"}, title="Distribution of total credit utilization (%)", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_util, util.to_frame(name="util_pct"))
        try:
            pct_over50 = (util > 50).mean() * 100
            show_question(2, "How many borrowers use >50% of their total credit limit?", f"**{pct_over50:.2f}%**")
        except Exception:
            pass
    else:
        st.write("No total credit limit data available to compute utilization.")

# Credit inquiries
with st.container():
    st.subheader("Credit inquiries (last 12 months)")
    if "inquiries_last_12m" in df.columns:
        fig_inq = None
        if _HAS_PLOTLY:
            fig_inq = px.histogram(df, x="inquiries_last_12m", nbins=20, title="Credit inquiries in last 12 months", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_inq, df[["inquiries_last_12m"]])
        pct_recent_shop = (df["inquiries_last_12m"] >= 3).mean() * 100
        show_question(3, "What percent have 3+ inquiries (active credit shopping)?", f"**{pct_recent_shop:.2f}%**")
    else:
        st.write("No inquiries_last_12m column available.")

# Open credit card accounts
with st.container():
    st.subheader("Open credit card accounts")
    if "num_open_cc_accounts" in df.columns:
        fig_cc = None
        if _HAS_PLOTLY:
            fig_cc = px.histogram(df, x="num_open_cc_accounts", nbins=20, title="Number of open credit card accounts", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_cc, df[["num_open_cc_accounts"]])
    else:
        st.write("No num_open_cc_accounts column available.")

# Delinquencies and tax liens
with st.container():
    st.subheader("Delinquencies & tax liens")
    col1, col2 = st.columns(2)
    with col1:
        if "delinq_2y" in df.columns:
            delinq_counts = df["delinq_2y"].value_counts().sort_index()
            fig_delinq = None
            if _HAS_PLOTLY:
                fig_delinq = px.bar(x=delinq_counts.index, y=delinq_counts.values, title="Delinquencies in last 2 years", **_PX_KWARGS)
            _show_plotly_or_fallback(fig_delinq, delinq_counts.reset_index().rename(columns={"index":"delinq_2y", "delinq_2y":"count"}))
            pct_delinq = (df["delinq_2y"] > 0).mean() * 100
            show_question(4, "What percent have at least one delinquency in 2 years?", f"**{pct_delinq:.2f}%**")
        else:
            st.write("No delinq_2y column available.")
    with col2:
        if "tax_liens" in df.columns:
            liens = (df["tax_liens"] > 0).value_counts()
            fig_liens = None
            if _HAS_PLOTLY:
                fig_liens = px.bar(x=liens.index.astype(str), y=liens.values, title="Has tax liens (True/False)", **_PX_KWARGS)
            _show_plotly_or_fallback(fig_liens, liens.reset_index().rename(columns={"index":"has_lien", "tax_liens":"count"}))
            pct_liens = (df["tax_liens"] > 0).mean() * 100
            show_question(5, "Percent with tax liens:", f"**{pct_liens:.2f}%**")
        else:
            st.write("No tax_liens column available.")

# Interest rate ECDF to inspect distribution tails
with st.container():
    st.subheader("Interest rate — ECDF")
    if "interest_rate" in df.columns:
        fig_ecdf = None
        if _HAS_PLOTLY:
            try:
                fig_ecdf = px.ecdf(df, x="interest_rate", title="ECDF of interest rates", **_PX_KWARGS)
            except Exception:
                fig_ecdf = None
        _show_plotly_or_fallback(fig_ecdf, df[["interest_rate"]])
    else:
        st.write("No interest_rate column available.")

# Top employers (long-tailed) — show top 15
with st.container():
    st.subheader("Top employer titles (top 15)")
    if "emp_title" in df.columns:
        top_jobs = df["emp_title"].fillna("(missing)").str.strip().str.lower().value_counts().nlargest(15)
        fig_jobs = None
        if _HAS_PLOTLY:
            fig_jobs = px.bar(x=top_jobs.index, y=top_jobs.values, title="Top 15 reported job titles", **_PX_KWARGS)
            fig_jobs.update_xaxes(tickangle=45)
        _show_plotly_or_fallback(fig_jobs, top_jobs.reset_index().rename(columns={"index":"emp_title", "emp_title":"count"}))
        show_question(6, "Do a few job titles dominate the dataset? Check if top titles are very common compared to the long tail.")
    else:
        st.write("No emp_title column available.")

# Grade vs interest (categorical univariate comparison)
with st.container():
    st.subheader("Loan grade vs interest rate")
    if "grade" in df.columns and "interest_rate" in df.columns:
        med_by_grade = df.groupby("grade")["interest_rate"].median().sort_index()
        fig_grade_ir = None
        if _HAS_PLOTLY:
            fig_grade_ir = px.bar(x=med_by_grade.index.astype(str), y=med_by_grade.values, title="Median interest rate by loan grade", labels={"x":"Grade","y":"Median interest rate (%)"}, **_PX_KWARGS)
        _show_plotly_or_fallback(fig_grade_ir, med_by_grade.reset_index().rename(columns={"grade":"grade","interest_rate":"median_ir"}))
        show_question(7, "Do higher grades (A) have meaningfully lower median interest rates than lower grades (G)?")
    else:
        st.write("Grade vs interest data not available.")

st.markdown("---")




# Term distribution
with st.container():
    st.subheader("Loan term distribution")
    if "term" in df.columns:
        term_counts = df["term"].value_counts().sort_index()
        fig_term = None
        if _HAS_PLOTLY:
            fig_term = px.bar(x=term_counts.index.astype(str), y=term_counts.values, title="Loan term counts", labels={"x":"Term (months)", "y":"Count"}, **_PX_KWARGS)
        _show_plotly_or_fallback(fig_term, term_counts.reset_index().rename(columns={"index":"term","term":"count"}))
        show_question(9, "Which loan term is most common? (36 vs 60 months)")
    else:
        st.write("No term column available.")

# Verified income vs interest rate
with st.container():
    st.subheader("Verified income vs Interest Rate")
    if "verified_income" in df.columns and "interest_rate" in df.columns:
        med_by_ver = df.groupby("verified_income")["interest_rate"].median().round(2).sort_index()
        fig_verified = None
        if _HAS_PLOTLY:
            fig_verified = px.bar(x=med_by_ver.index.astype(str), y=med_by_ver.values, title="Median interest rate by income verification status", labels={"x":"Verified income","y":"Median interest rate (%)"}, **_PX_KWARGS)
        _show_plotly_or_fallback(fig_verified, med_by_ver.reset_index().rename(columns={"verified_income":"verified_income","interest_rate":"median_ir"}))
        show_question(10, "Do verified borrowers get lower interest rates?", f"Median rates: {med_by_ver.to_dict()}")
    else:
        st.write("No verified_income column available.")

# Initial listing status and disbursement method
with st.container():
    st.subheader("Initial listing & disbursement")
    if "initial_listing_status" in df.columns:
        ils = df["initial_listing_status"].value_counts()
        fig_ils = None
        if _HAS_PLOTLY:
            fig_ils = px.bar(x=ils.index, y=ils.values, title="Initial listing status", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_ils, ils.reset_index().rename(columns={"index":"initial_listing_status","initial_listing_status":"count"}))
    else:
        st.write("No initial_listing_status column available.")

    if "disbursement_method" in df.columns:
        dm = df["disbursement_method"].value_counts()
        fig_dm = None
        if _HAS_PLOTLY:
            fig_dm = px.pie(values=dm.values, names=dm.index, title="Disbursement method", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_dm, dm.reset_index().rename(columns={"index":"disbursement_method","disbursement_method":"count"}))

# Balance distribution
with st.container():
    st.subheader("Current balance distribution")
    if "balance" in df.columns:
        fig_bal = None
        if _HAS_PLOTLY:
            fig_bal = px.histogram(df, x="balance", nbins=50, title="Outstanding balance distribution", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_bal, df[["balance"]])
        try:
            pct_zero_bal = (df["balance"] == 0).mean() * 100
            show_question(11, "Percent with zero outstanding balance:", f"**{pct_zero_bal:.2f}%**")
        except Exception:
            pass
    else:
        st.write("No balance column available.")

# Number of credit lines and cards carrying balance
with st.container():
    st.subheader("Credit lines & cards carrying balance")
    if "total_credit_lines" in df.columns:
        fig_lines = None
        if _HAS_PLOTLY:
            fig_lines = px.histogram(df, x="total_credit_lines", nbins=40, title="Total credit lines distribution", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_lines, df[["total_credit_lines"]])
    else:
        st.write("No total_credit_lines column available.")

    if "num_cc_carrying_balance" in df.columns:
        fig_ccbal = None
        if _HAS_PLOTLY:
            fig_ccbal = px.histogram(df, x="num_cc_carrying_balance", nbins=20, title="Number of credit cards carrying balance", **_PX_KWARGS)
        _show_plotly_or_fallback(fig_ccbal, df[["num_cc_carrying_balance"]])

st.markdown("---")
st.markdown("### Notes")
st.write("This page shows many univariate distributions, KPIs, and short analysis questions derived from `cleaned_df.csv`. Use the sidebar to tweak histogram bins and toggle log-scale for income. If you'd like additional specific charts or downloadable tables/plots, tell me which ones and I'll add them.")
