import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False
import numpy as np
from typing import Optional


def _show_plotly_or_fallback(fig, fallback_df=None):
    """Show a plotly figure if available, otherwise show a small fallback table and message."""
    if _HAS_PLOTLY and fig is not None:
        try:
            fig.update_layout(colorway=PASTEL_PALETTE)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly is not installed. Install with: pip install plotly to see interactive charts.")
        if fallback_df is not None:
            try:
                st.dataframe(fallback_df.head(100))
            except Exception:
                st.write("Preview not available.")



def show_question(n: int, question: str, answer: Optional[str] = None):
    """Render a prominent question and optional answer."""
    st.info(f"Question {n}: {question}")
    if answer is not None:
        st.markdown(f"**Answer:** {answer}")


# Default Plotly styling kwargs for readable, colorful charts (module-level)
PASTEL_PALETTE = [
    "#AEC6CF",  # soft blue-gray
    "#FFB7B2",  # pastel pink
    "#FDFD96",  # pastel yellow
    "#B39EB5",  # pastel purple
    "#77DD77",  # pastel green
    "#CFCFC4",  # light gray
    "#FFD1DC",  # light rose
    "#B5EAD7",  # mint
]

if _HAS_PLOTLY:
    # Use a uniform pastel palette for readability and consistent look across pages.
    _PX_KWARGS = {"template": "plotly_white", "color_discrete_sequence": PASTEL_PALETTE}
else:
    _PX_KWARGS = {"template": "plotly_white"}


def _px_kwargs_for(kind: str = "default"):
    """Return a filtered copy of _PX_KWARGS appropriate for specific px functions.

    Some px functions (e.g., px.imshow) do not accept certain kwargs like
    `color_discrete_sequence`. This helper ensures we only pass supported
    args for those functions while keeping a consistent template elsewhere.
    """
    if not _HAS_PLOTLY:
        return {}
    if kind == "imshow":
        # px.imshow accepts template but not color_discrete_sequence. If a
        # continuous color scale were needed, add 'color_continuous_scale' to
        # _PX_KWARGS and allow it here. For now, restrict to template only.
        return {k: v for k, v in _PX_KWARGS.items() if k in ("template", "color_continuous_scale")}
    return dict(_PX_KWARGS)

st.set_page_config(layout="wide")
st.title("Multivariate Analysis — Cleaned Loans Dataset")


def render_metric(label: str, value: str):
        """Render a consistent metric with HTML so font sizes are uniform across pages."""
        html = f"""
        <div style='line-height:1.1; margin-bottom:6px;'>
            <div style='font-size:13px; color:#6b6b6b;'>{label}</div>
            <div style='font-size:20px; font-weight:700; color:#111;'>{value}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path, index_col=0)

DATA_PATH = "cleaned_df.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"{DATA_PATH} not found. Place the file in the project root.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# (Removed older Key dataset KPIs to avoid duplication — Key Metrics Cards below remain)

# --- Key Metrics Cards (compact, at-a-glance metrics) -----------------
st.markdown("## Key Metrics Cards")
try:
    # First row of 4 cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Total loans", f"{df.shape[0]:,}")
    with c2:
        try:
            avg_loan = df["loan_amount"].dropna().mean()
            render_metric("Avg loan amount", f"${avg_loan:,.0f}")
        except Exception:
            render_metric("Avg loan amount", "N/A")
    with c3:
        try:
            med_loan = df["loan_amount"].dropna().median()
            render_metric("Median loan", f"${med_loan:,.0f}")
        except Exception:
            render_metric("Median loan", "N/A")
    with c4:
        try:
            avg_ir = df["interest_rate"].dropna().mean()
            render_metric("Avg interest rate", f"{avg_ir:.2f}%")
        except Exception:
            render_metric("Avg interest rate", "N/A")

    # Second row of 4 cards
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        try:
            if "loan_status" in df.columns:
                charged = df["loan_status"].astype(str).str.contains("Charged|Default|charged|default", case=False, na=False).mean() * 100
                render_metric("% charged-off/default", f"{charged:.2f}%")
            else:
                render_metric("% charged-off/default", "N/A")
        except Exception:
            render_metric("% charged-off/default", "N/A")
    with c6:
        try:
            avg_dti = df["debt_to_income"].dropna().mean()
            render_metric("Avg DTI (%)", f"{avg_dti:.2f}%")
        except Exception:
            render_metric("Avg DTI (%)", "N/A")
    with c7:
        try:
            if ("total_credit_limit" in df.columns) and ("total_credit_utilized" in df.columns):
                util = (df["total_credit_utilized"] / df["total_credit_limit"]).replace([np.inf, -np.inf], np.nan) * 100
                render_metric("Avg credit utilization", f"{util.dropna().mean():.2f}%")
            else:
                render_metric("Avg credit utilization", "N/A")
        except Exception:
            render_metric("Avg credit utilization", "N/A")
    with c8:
        try:
            if ("installment" in df.columns) and ("annual_income" in df.columns):
                monthly_income = df["annual_income"] / 12
                pay_pct = (df["installment"] / monthly_income) * 100
                pay_pct = pay_pct.replace([np.inf, -np.inf], np.nan)
                render_metric("Avg payment burden (%)", f"{pay_pct.dropna().mean():.2f}%")
            else:
                render_metric("Avg payment burden (%)", "N/A")
        except Exception:
            render_metric("Avg payment burden (%)", "N/A")
except Exception:
    st.write("Key metrics unavailable due to missing columns or data errors.")

st.markdown("## Multivariate charts and short analysis questions")

# Sidebar controls
st.sidebar.header("Multivariate controls")
grade_filter = st.sidebar.multiselect("Filter grades (leave empty for all)", options=sorted(df["grade"].dropna().unique()))
term_filter = st.sidebar.multiselect("Filter terms", options=sorted(df["term"].dropna().unique()))
sample_frac = st.sidebar.slider("Sample fraction for heavy plots (scatter)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)

plot_df = df.copy()
if grade_filter:
    plot_df = plot_df[plot_df["grade"].isin(grade_filter)]
if term_filter:
    plot_df = plot_df[plot_df["term"].isin(term_filter)]

# 1) Annual income vs Loan amount (scatter)
st.subheader("Annual income vs Loan amount")
sample = plot_df.sample(frac=min(sample_frac, 1.0), random_state=1) if len(plot_df) > 100 else plot_df
if _HAS_PLOTLY:
    fig1 = px.scatter(sample, x="annual_income", y="loan_amount", color="grade", hover_data=["emp_title","state","loan_purpose"],
                      title="Loan amount by annual income (colored by grade)", opacity=0.6, **_PX_KWARGS)
else:
    fig1 = None
_show_plotly_or_fallback(fig1, fallback_df=sample)
# Simple aggregated view: median loan amount per income bucket
income_bins = pd.qcut(plot_df["annual_income"].dropna(), q=6, duplicates='drop')
median_loan_by_income = plot_df.dropna(subset=["annual_income","loan_amount"]).groupby(income_bins)["loan_amount"].median().reset_index()
median_loan_by_income.columns = ["income_bin","median_loan_amount"]
# Convert Interval objects to strings so plotting libraries and JSON serializers can handle them
median_loan_by_income["income_bin"] = median_loan_by_income["income_bin"].astype(str)
if _HAS_PLOTLY:
    fig1b = px.bar(median_loan_by_income, x="income_bin", y="median_loan_amount", title="Median loan amount by income bin", **_PX_KWARGS)
    fig1b.update_xaxes(tickangle=45)
else:
    fig1b = None
_show_plotly_or_fallback(fig1b, fallback_df=median_loan_by_income)
show_question(1, "Does higher income always imply larger loans? Look at the median loan per income bin above to judge the trend.")

# 2) Interest rate vs Debt-to-Income (scatter + correlation)
st.subheader("Interest rate vs Debt-to-Income (DTI)")
clean = plot_df.dropna(subset=["interest_rate","debt_to_income"])
if clean.shape[0] > 0:
    sample2 = clean.sample(frac=min(sample_frac, 1.0), random_state=2) if len(clean) > 200 else clean
    if _HAS_PLOTLY:
        fig2 = px.scatter(sample2, x="debt_to_income", y="interest_rate", color="term", opacity=0.6,
                          title="Interest rate by Debt-to-Income (colored by term)", **_PX_KWARGS)
    else:
        fig2 = None
    _show_plotly_or_fallback(fig2, fallback_df=sample2)
    corr = clean[["interest_rate","debt_to_income"]].corr().iloc[0,1]
    show_question(2, "Is DTI correlated with interest rate?", f"Pearson r = **{corr:.2f}** (positive means higher DTI tends to have higher rates)")
else:
    st.write("Not enough data for Interest vs DTI analysis.")

# 3) Credit utilization vs Interest — compute utilization and show grouped medians
st.subheader("Credit utilization vs Interest rate")
if ("total_credit_limit" in plot_df.columns) and ("total_credit_utilized" in plot_df.columns):
    util = (plot_df["total_credit_utilized"] / plot_df["total_credit_limit"]) * 100
    plot_df = plot_df.assign(credit_utilization_pct=util)
    clean_util = plot_df.dropna(subset=["credit_utilization_pct","interest_rate"]) 
    if clean_util.shape[0] > 0:
        # scatter
        sample3 = clean_util.sample(frac=min(sample_frac, 1.0), random_state=3) if len(clean_util) > 200 else clean_util
        if _HAS_PLOTLY:
            fig3 = px.scatter(sample3, x="credit_utilization_pct", y="interest_rate", color="grade", opacity=0.6, title="Interest rate vs credit utilization %", **_PX_KWARGS)
        else:
            fig3 = None
        _show_plotly_or_fallback(fig3, fallback_df=sample3)
        # grouped medians
        clean_util["util_bucket"] = pd.cut(clean_util["credit_utilization_pct"].fillna(0), bins=[-1,10,30,50,70,100,1000], labels=["0-10%","10-30%","30-50%","50-70%","70-100%","100%+"])
        med_by_util = clean_util.groupby("util_bucket")["interest_rate"].median().reset_index()
        if _HAS_PLOTLY:
            fig3b = px.bar(med_by_util, x="util_bucket", y="interest_rate", title="Median interest rate by utilization bucket", labels={"util_bucket":"Utilization bucket","interest_rate":"Median interest rate"}, **_PX_KWARGS)
        else:
            fig3b = None
        _show_plotly_or_fallback(fig3b, fallback_df=med_by_util)
        show_question(3, "Do borrowers who use a larger share of their credit get higher interest rates? Inspect the bar chart of medians per utilization bucket.")
    else:
        st.write("Not enough utilization data to analyze.")
else:
    st.write("Credit limit/utilized columns not available to compute utilization.")

# 4) Grade vs charged-off rate (simple default proxy) — percent charged off by grade
st.subheader("Loan grade vs charged-off rate")
if "loan_status" in plot_df.columns:
    # define charged off statuses
    charged_mask = plot_df["loan_status"].astype(str).str.contains("Charged|Default|charged|default", case=False, na=False)
    grade_charged = plot_df.groupby("grade").apply(lambda g: charged_mask[g.index].mean() if len(g)>0 else np.nan).reset_index()
    grade_charged.columns = ["grade","pct_charged_off"]
    grade_charged["pct_charged_off"] = grade_charged["pct_charged_off"] * 100
    grade_charged = grade_charged.dropna().sort_values("grade")
    if not grade_charged.empty:
        if _HAS_PLOTLY:
            fig4 = px.bar(grade_charged, x="grade", y="pct_charged_off", title="Percent charged-off by grade", labels={"pct_charged_off":"% charged-off"}, **_PX_KWARGS)
        else:
            fig4 = None
        _show_plotly_or_fallback(fig4, fallback_df=grade_charged)
        show_question(4, "Do lower grades show higher charged-off rates? Check the percent charged-off per grade above.")
    else:
        st.write("No grade/loan_status data to compute charged-off rates.")
else:
    st.write("No loan_status column available to compute charged-off rates.")

# 5) Correlation heatmap (numeric columns)
st.subheader("Correlation matrix (numeric variables)")
num = plot_df.select_dtypes(include=[np.number])
if num.shape[1] >= 2:
    corr = num.corr().round(2)
    # limit to top correlated columns for readability (optional)
    if _HAS_PLOTLY:
        # px.imshow does not accept some discrete-color kwargs; use a filtered set
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (numeric)", **_px_kwargs_for("imshow"))
    else:
        fig_corr = None
    _show_plotly_or_fallback(fig_corr, fallback_df=corr)
    # find top absolute correlations (excluding self)
    corr_unstack = corr.abs().unstack().reset_index()
    corr_unstack.columns = ["var1","var2","abs_corr"]
    corr_unstack = corr_unstack[corr_unstack["var1"] != corr_unstack["var2"]]
    top_pairs = corr_unstack.sort_values("abs_corr", ascending=False).drop_duplicates(subset=["abs_corr"]).head(5)
    st.markdown("- Top absolute correlations (var1, var2, |r|):")
    for _, r in top_pairs.iterrows():
        st.write(f"  - {r['var1']} vs {r['var2']}: **{r['abs_corr']:.2f}**")
else:
    st.write("Not enough numeric columns to compute correlation matrix.")

# -----------------------------
# Additional multivariate questions (5-15)
# -----------------------------
st.header("Further multivariate explorations & clear questions")

# 5) Grade vs median loan amount
if ("grade" in plot_df.columns) and ("loan_amount" in plot_df.columns):
    med_loan_grade = plot_df.dropna(subset=["grade","loan_amount"]).groupby("grade")["loan_amount"].median().reset_index()
    med_loan_grade["grade"] = med_loan_grade["grade"].astype(str)
    if _HAS_PLOTLY:
        fig5 = px.bar(med_loan_grade, x="grade", y="loan_amount", title="Median loan amount by grade", labels={"loan_amount":"Median loan amount"}, **_PX_KWARGS)
    else:
        fig5 = None
    _show_plotly_or_fallback(fig5, fallback_df=med_loan_grade)
    show_question(5, "How does median loan amount vary by loan grade?", "Check the bar chart for grade-level medians.")

# 6) Sub-grade vs median interest (top sub-grades)
if ("sub_grade" in plot_df.columns) and ("interest_rate" in plot_df.columns):
    med_sub = plot_df.dropna(subset=["sub_grade","interest_rate"]).groupby("sub_grade")["interest_rate"].median().reset_index()
    med_sub = med_sub.sort_values("interest_rate").head(20)
    med_sub["sub_grade"] = med_sub["sub_grade"].astype(str)
    if _HAS_PLOTLY:
        fig6 = px.bar(med_sub, x="sub_grade", y="interest_rate", title="Median interest rate by sub-grade (top 20)", **_PX_KWARGS)
        fig6.update_xaxes(tickangle=45)
    else:
        fig6 = None
    _show_plotly_or_fallback(fig6, fallback_df=med_sub)
    show_question(6, "Which sub-grades have the highest median interest rates?", "Look at the top 20 sub-grades by median interest rate.")

# 7) Loan purpose vs charged-off rate
if ("loan_purpose" in plot_df.columns) and ("loan_status" in plot_df.columns):
    charged_mask = plot_df["loan_status"].astype(str).str.contains("Charged|Default|charged|default", case=False, na=False)
    purpose_charged = plot_df.groupby("loan_purpose").apply(lambda g: charged_mask[g.index].mean() if len(g)>0 else np.nan).reset_index()
    purpose_charged.columns = ["loan_purpose","pct_charged_off"]
    purpose_charged = purpose_charged.dropna().sort_values("pct_charged_off", ascending=False).head(20)
    purpose_charged["pct_charged_off"] = purpose_charged["pct_charged_off"] * 100
    if _HAS_PLOTLY:
        fig7 = px.bar(purpose_charged, x="loan_purpose", y="pct_charged_off", title="Percent charged-off by loan purpose (top 20)", **_PX_KWARGS)
        fig7.update_xaxes(tickangle=45)
    else:
        fig7 = None
    _show_plotly_or_fallback(fig7, fallback_df=purpose_charged)
    show_question(7, "Which loan purposes have higher charged-off rates?", "Inspect the top 20 loan purposes by percent charged-off.")

# 8) State-level median income (top states by count)
if ("state" in plot_df.columns) and ("annual_income" in plot_df.columns):
    state_counts = plot_df["state"].value_counts().nlargest(12).index.tolist()
    state_income = plot_df[plot_df["state"].isin(state_counts)].groupby("state")["annual_income"].median().reset_index().sort_values("annual_income", ascending=False)
    if _HAS_PLOTLY:
        fig8 = px.bar(state_income, x="state", y="annual_income", title="Median annual income for top states (by count)", **_PX_KWARGS)
    else:
        fig8 = None
    _show_plotly_or_fallback(fig8, fallback_df=state_income)
    show_question(8, "Which states have the highest median incomes among the top borrower states?", "Check the bar chart for median income by state.")

# 9) Term vs charged-off percent and median interest
if ("term" in plot_df.columns) and ("loan_status" in plot_df.columns):
    term_charged = plot_df.groupby("term").apply(lambda g: charged_mask[g.index].mean() if len(g)>0 else np.nan).reset_index()
    term_charged.columns = ["term","pct_charged_off"]
    term_charged["pct_charged_off"] = term_charged["pct_charged_off"] * 100
    term_charged = term_charged.dropna().sort_values("term")
    term_interest = plot_df.groupby("term")["interest_rate"].median().reset_index()
    term_combo = term_charged.merge(term_interest, on="term", how="left")
    if _HAS_PLOTLY:
        fig9 = px.bar(term_combo, x="term", y="pct_charged_off", title="Percent charged-off by term", **_PX_KWARGS)
    else:
        fig9 = None
    _show_plotly_or_fallback(fig9, fallback_df=term_combo)
    show_question(9, "Do longer-term loans (e.g., 60 months) have higher charged-off rates or different median interest rates?", "See the charged-off percent by term.")

# 10) (Removed) Income-to-loan ratio analysis removed per request
# 11) Installment vs interest rate (are higher rates linked to higher payments?)
if ("installment" in plot_df.columns) and ("interest_rate" in plot_df.columns):
    inst_clean = plot_df.dropna(subset=["installment","interest_rate"]) 
    sample_inst = inst_clean.sample(frac=min(sample_frac,1.0), random_state=11) if len(inst_clean)>200 else inst_clean
    if _HAS_PLOTLY:
        fig11 = px.scatter(sample_inst, x="installment", y="interest_rate", title="Installment vs interest rate", opacity=0.6, **_PX_KWARGS)
    else:
        fig11 = None
    _show_plotly_or_fallback(fig11, fallback_df=sample_inst)
    show_question(11, "Do higher monthly installments correspond to higher interest rates?", "Inspect the scatter of installment vs interest rate.")

# 12) Delinquencies vs credit utilization (binned medians)
if ("delinq_2y" in plot_df.columns) and ("credit_utilization_pct" in plot_df.columns):
    dq = plot_df.dropna(subset=["delinq_2y","credit_utilization_pct"]) 
    if not dq.empty:
        dq["util_bin"] = pd.cut(dq["credit_utilization_pct"].fillna(0), bins=[-1,10,30,50,70,100,1000], labels=["0-10%","10-30%","30-50%","50-70%","70-100%","100%+"])
        med_delinq = dq.groupby("util_bin")["delinq_2y"].mean().reset_index()
        med_delinq["util_bin"] = med_delinq["util_bin"].astype(str)
        if _HAS_PLOTLY:
            fig12 = px.bar(med_delinq, x="util_bin", y="delinq_2y", title="Average delinquencies by utilization bin", labels={"delinq_2y":"Avg delinquencies"}, **_PX_KWARGS)
        else:
            fig12 = None
        _show_plotly_or_fallback(fig12, fallback_df=med_delinq)
        show_question(12, "Are higher utilization borrowers more likely to have delinquencies?", "Check average delinquencies per utilization bin.")

# 13) Issue month trends (loan counts and median interest by issue_month)
if "issue_month" in plot_df.columns:
    im = plot_df.dropna(subset=["issue_month"]) 
    # Build counts reliably and avoid duplicate column names
    im_counts = im["issue_month"].value_counts().nlargest(20).rename_axis("issue_month").reset_index(name="count")
    im_med = im.groupby("issue_month")["interest_rate"].median().reset_index().sort_values("issue_month")
    if _HAS_PLOTLY:
        fig13 = px.bar(im_counts, x="issue_month", y="count", title="Top issue months by loan count (top 20)", **_PX_KWARGS)
        fig13.update_xaxes(tickangle=45)
    else:
        fig13 = None
    _show_plotly_or_fallback(fig13, fallback_df=im_counts)
    show_question(13, "Which months had the most loans issued?", "Check the top issue months by count.")

# 14) Top employers — median loan amount (top 15)
if "emp_title" in plot_df.columns and "loan_amount" in plot_df.columns:
    job_series = plot_df["emp_title"].fillna("(missing)").astype(str).str.strip().str.lower()
    top_jobs = job_series.value_counts().nlargest(15).index.tolist()
    job_med = plot_df.assign(job=job_series).loc[plot_df.assign(job=job_series)["job"].isin(top_jobs)].groupby("job")["loan_amount"].median().reset_index().rename(columns={"loan_amount":"median_loan"}).sort_values("median_loan", ascending=False)
    if _HAS_PLOTLY:
        fig14 = px.bar(job_med, x="job", y="median_loan", title="Median loan amount for top employers", **_PX_KWARGS)
        fig14.update_xaxes(tickangle=45)
    else:
        fig14 = None
    _show_plotly_or_fallback(fig14, fallback_df=job_med)
    show_question(14, "Do specific employers take larger loans on average?", "See median loan by top reported employer titles.")

# 15) Quick scatter for top correlated variable pair beyond self-correlation
if num.shape[1] >= 2:
    top_pairs = corr_unstack.sort_values("abs_corr", ascending=False).drop_duplicates(subset=["abs_corr"]).head(10)
    # pick the top pair where var1 != var2
    pair = None
    for _, r in top_pairs.iterrows():
        if r["var1"] != r["var2"]:
            pair = (r["var1"], r["var2"], r["abs_corr"])
            break
    if pair is not None:
        v1, v2, strength = pair
        scatter_df = plot_df.dropna(subset=[v1, v2])
        sample_pair = scatter_df.sample(frac=min(sample_frac,1.0), random_state=15) if len(scatter_df)>200 else scatter_df
        if _HAS_PLOTLY:
            fig15 = px.scatter(sample_pair, x=v1, y=v2, title=f"Scatter: {v1} vs {v2} (|r|={strength:.2f})", **_PX_KWARGS)
        else:
            fig15 = None
        _show_plotly_or_fallback(fig15, fallback_df=sample_pair)
        show_question(15, f"Inspect the relationship between {v1} and {v2}", f"Absolute correlation |r| = {strength:.2f}")

st.markdown("---")
st.write("Notes: These charts show relationships between key variables. If you'd like formal statistical tests (regression) or downloadable result tables, I can add them.")
