import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# Plotly optional
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False

# Plotly defaults and helper
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
            fig.update_layout(colorway=PASTEL_PALETTE)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not available — showing data preview instead.")
        if fallback_df is not None:
            st.dataframe(fallback_df.head(50))


st.set_page_config(layout="wide")
st.title("Borrower Profile")


def render_metric(label: str, value: str):
        html = f"""
        <div style='line-height:1.1; margin-bottom:6px;'>
            <div style='font-size:13px; color:#6b6b6b;'>{label}</div>
            <div style='font-size:20px; font-weight:700; color:#111;'>{value}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def load_data(path: str = "cleaned_df.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

# Load dataset
DATA_PATH = "cleaned_df.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"{DATA_PATH} not found. Put the file in the project root.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# Sidebar controls: select by index or search by employer
st.sidebar.header("Borrower selector")
select_mode = st.sidebar.radio("Select mode:", ("By index", "By employer (top 50)", "Random sample"))
selected_idx = None

if select_mode == "By index":
    # Show a text input and a selectbox with a small sample for convenience
    idx_input = st.sidebar.text_input("Enter borrower index (exact)", value=str(df.index[0]))
    try:
        # index may be integer or string; coerce carefully
        if idx_input != "":
            if df.index.dtype.kind in ("i", "u"):
                selected_idx = int(idx_input)
            else:
                selected_idx = idx_input
    except Exception:
        selected_idx = None
    # quick select of recent indices
    pick = st.sidebar.selectbox("Or pick from first 200 rows:", options=[int(i) for i in df.index[:200]])
    if st.sidebar.button("Use selected pick"):
        selected_idx = pick

elif select_mode == "By employer (top 50)":
    top_emps = df["emp_title"].fillna("(missing)").str.strip().str.lower().value_counts().nlargest(50).index.tolist()
    chosen = st.sidebar.selectbox("Top employers:", options=top_emps)
    if chosen:
        # pick the first matching borrower for that employer
        matches = df[df["emp_title"].fillna("(missing)").str.strip().str.lower() == chosen]
        if not matches.empty:
            selected_idx = matches.index[0]

else:
    sample_n = st.sidebar.slider("Sample size", min_value=1, max_value=20, value=5)
    sample_idxs = df.sample(n=sample_n, random_state=1).index.tolist()
    selected_idx = st.sidebar.selectbox("Random sample pick:", options=sample_idxs)

if selected_idx is None or selected_idx not in df.index:
    st.info("Select a borrower from the sidebar to view a detailed profile.")
    st.stop()

# Retrieve borrower row (as Series)
borrower = df.loc[selected_idx]

# Top-level metrics for this borrower
st.header(f"Borrower profile — index: {selected_idx}")
col_a, col_b = st.columns([2, 3])
with col_a:
    # Core numeric KPIs
    try:
        income = float(borrower.get("annual_income", np.nan))
    except Exception:
        income = np.nan
    try:
        loan_amt = float(borrower.get("loan_amount", np.nan))
    except Exception:
        loan_amt = np.nan
    try:
        ir = float(borrower.get("interest_rate", np.nan))
    except Exception:
        ir = np.nan
    try:
        term = borrower.get("term", "N/A")
    except Exception:
        term = "N/A"

    # KPIs as metrics
    k1, k2, k3 = st.columns(3)
    with k1:
        try:
            if not np.isnan(income):
                render_metric("Annual income", f"${income:,.0f}")
            else:
                render_metric("Annual income", "N/A")
        except Exception:
            render_metric("Annual income", "N/A")
    with k2:
        try:
            if not np.isnan(loan_amt):
                render_metric("Loan amount", f"${loan_amt:,.0f}")
            else:
                render_metric("Loan amount", "N/A")
        except Exception:
            render_metric("Loan amount", "N/A")
    with k3:
        try:
            if not np.isnan(ir):
                render_metric("Interest rate", f"{ir:.2f}%")
            else:
                render_metric("Interest rate", "N/A")
        except Exception:
            render_metric("Interest rate", "N/A")

    # Derived metrics
    try:
        installment = float(borrower.get("installment", np.nan))
        if not np.isnan(income) and income > 0:
            monthly_income = income / 12.0
            payment_burden = (installment / monthly_income) * 100
        else:
            payment_burden = np.nan
    except Exception:
        installment = np.nan
        payment_burden = np.nan

    try:
        if ("total_credit_limit" in df.columns) and ("total_credit_utilized" in df.columns):
            util_pct = (borrower.get("total_credit_utilized", np.nan) / borrower.get("total_credit_limit", np.nan)) * 100
        else:
            util_pct = np.nan
    except Exception:
        util_pct = np.nan

    c1, c2 = st.columns(2)
    with c1:
        if not np.isnan(payment_burden):
            render_metric("Payment burden (installment / monthly income)", f"{payment_burden:.2f}%")
        else:
            render_metric("Payment burden (installment / monthly income)", "N/A")
    with c2:
        if not np.isnan(util_pct):
            render_metric("Credit utilization", f"{util_pct:.2f}%")
        else:
            render_metric("Credit utilization", "N/A")

    # Basic demographics / employment
    st.markdown("**Demographics & employment**")
    st.write(f"- State: **{borrower.get('state','N/A')}**")
    st.write(f"- Homeownership: **{borrower.get('homeownership','N/A')}**")
    st.write(f"- Employer / title: **{borrower.get('emp_title','(missing)')}**")

with col_b:
    # Loan details and status
    st.markdown("**Loan details**")
    st.write(f"- Purpose: **{borrower.get('loan_purpose','N/A')}**")
    st.write(f"- Term: **{borrower.get('term','N/A')} months**")
    st.write(f"- Grade / sub-grade: **{borrower.get('grade','N/A')} / {borrower.get('sub_grade','N/A')}**")
    st.write(f"- Loan status: **{borrower.get('loan_status','N/A')}**")
    st.write(f"- Disbursement method: **{borrower.get('disbursement_method','N/A')}**")
    st.write(f"- Has delinquency: **{borrower.get('has_delinquency','N/A')}**")

# Comparison charts: income vs dataset and grade distribution
st.subheader("How this borrower compares to the dataset")
comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    # Annual income distribution with vertical marker
    if "annual_income" in df.columns:
        try:
            inc_series = df["annual_income"].dropna()
            if _HAS_PLOTLY:
                fig = px.histogram(inc_series, nbins=40, title="Annual income distribution (dataset)", labels={"value":"Annual income"}, **_PX_KWARGS)
                if not np.isnan(income):
                    fig.add_vline(x=income, line_dash="dash", line_color=PASTEL_PALETTE[1], annotation_text="Selected borrower", annotation_position="top right")
                _show_plotly_or_fallback(fig, fallback_df=pd.DataFrame({"annual_income": inc_series}))
            else:
                st.write("Annual income distribution")
                st.dataframe(inc_series.describe())
        except Exception as e:
            st.write("Income visualization not available.")

with comp_col2:
    if "grade" in df.columns:
        try:
            grade_counts = df["grade"].fillna("(missing)").value_counts().sort_index()
            if _HAS_PLOTLY:
                figg = px.bar(x=grade_counts.index.astype(str), y=grade_counts.values, title="Grade distribution (dataset)", labels={"x":"Grade","y":"Count"}, **_PX_KWARGS)
                # annotate the borrower's grade
                try:
                    bgrade = str(borrower.get("grade", "(missing)"))
                    # highlight by adding an annotation
                    figg.add_vline(x=list(grade_counts.index).index(bgrade) if bgrade in list(grade_counts.index) else 0,
                                   line_color=PASTEL_PALETTE[1], line_dash="dash")
                except Exception:
                    pass
                _show_plotly_or_fallback(figg, fallback_df=pd.DataFrame({"grade": grade_counts.index, "count": grade_counts.values}))
            else:
                st.dataframe(grade_counts)
        except Exception:
            st.write("Grade distribution not available.")

# Credit history snapshot
st.subheader("Credit history snapshot")
ch_col1, ch_col2, ch_col3 = st.columns(3)
with ch_col1:
    st.write(f"- Delinquencies (2y): **{borrower.get('delinq_2y','N/A')}**")
    st.write(f"- Inquiries (12m): **{borrower.get('inquiries_last_12m','N/A')}**")
with ch_col2:
    st.write(f"- Tax liens: **{borrower.get('tax_liens','N/A')}**")
    st.write(f"- Public bankrupt: **{borrower.get('public_record_bankrupt','N/A')}**")
with ch_col3:
    st.write(f"- Total credit limit: **{borrower.get('total_credit_limit','N/A')}**")
    st.write(f"- Total credit utilized: **{borrower.get('total_credit_utilized','N/A')}**")

# Raw borrower record and download
st.subheader("Raw borrower record")
try:
    st.dataframe(pd.DataFrame(borrower).T)
except Exception:
    st.write(borrower)

# Download individual borrower row as CSV
try:
    row_csv = pd.DataFrame(borrower).T.to_csv(index=True)
    st.download_button("Download borrower record (CSV)", data=row_csv, file_name=f"borrower_{selected_idx}.csv", mime="text/csv")
except Exception:
    st.write("Download not available.")

# Quick suggestions / follow-ups
st.markdown("---")
st.markdown("Suggestions: compare this borrower to their grade cohort or to borrowers in the same state. You can also explore payment history in the raw data if available.")
