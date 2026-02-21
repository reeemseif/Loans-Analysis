
import streamlit as st
import pandas as pd
import json
import os
import numpy as np

try:
	import plotly.express as px
	_HAS_PLOTLY = True
except Exception:
	px = None
	_HAS_PLOTLY = False


# Human-friendly descriptions for every column in cleaned_df.csv
DESCRIPTIONS = {
	"emp_title": "Employee job title as reported by the borrower.",
	"experience_years": "Years of work experience (numeric).",
	"state": "U.S. state of residence (two-letter code).",
	"homeownership": "Homeownership status (e.g., RENT, MORTGAGE, OWN).",
	"annual_income": "Reported annual income in US dollars.",
	"verified_income": "Income verification status (Verified or Not Verified).",
	"debt_to_income": "Debt-to-income (DTI) ratio (percentage).",
	"delinq_2y": "Number of delinquencies in the past 2 years.",
	"earliest_credit_line": "Year or date of the borrower's earliest reported credit line.",
	"inquiries_last_12m": "Number of credit inquiries in the last 12 months.",
	"total_credit_lines": "Total number of credit lines reported for the borrower.",
	"open_credit_lines": "Number of currently open credit lines.",
	"total_credit_limit": "Total credit limit across all reporting accounts (dollars).",
	"total_credit_utilized": "Total credit amount currently utilized (dollars).",
	"num_collections_last_12m": "Number of collection accounts opened in the last 12 months.",
	"total_collection_amount_ever": "Total amount ever sent to collections (dollars).",
	"num_open_cc_accounts": "Number of open credit card accounts.",
	"num_cc_carrying_balance": "Number of credit cards that currently carry a balance.",
	"tax_liens": "Number of tax liens on the borrower's record.",
	"public_record_bankrupt": "Count of public-record bankruptcies for the borrower.",
	"loan_purpose": "Stated purpose for the loan (e.g., debt_consolidation, moving).",
	"application_type": "Type of application (individual or joint).",
	"loan_amount": "Requested loan amount in US dollars.",
	"term": "Loan term in months (e.g., 36 or 60).",
	"interest_rate": "Annual interest rate for the loan (percentage).",
	"installment": "Monthly payment amount (dollars).",
	"grade": "Loan grade assigned by the platform/lender (A-G).",
	"sub_grade": "More granular loan sub-grade under the main grade.",
	"issue_month": "Month and year the loan was issued (e.g., Mar-2018).",
	"loan_status": "Current status of the loan (Current, Fully Paid, Charged Off, etc.).",
	"initial_listing_status": "Initial listing status when loan was posted on the platform.",
	"disbursement_method": "Method used to disburse the loan funds (e.g., Cash).",
	"balance": "Current outstanding principal balance (dollars).",
	"has_balance": "Boolean flag indicating if the borrower currently has a non-zero balance.",
	"has_tax_lien": "Boolean flag indicating whether the borrower has any tax liens.",
}


def summarize_column(col_series: pd.Series) -> dict:
	"""Return a summary dict for a pandas Series usable for UI and export."""
	s = col_series
	info = {
		"dtype": str(s.dtype),
		"missing_count": int(s.isna().sum()),
		"missing_pct": float(round(s.isna().mean() * 100, 2)),
		"unique_count": int(s.nunique(dropna=True)),
		"sample_values": []
	}

	try:
		if pd.api.types.is_numeric_dtype(s):
			info.update({
				"min": float(s.min()) if not s.dropna().empty else None,
				"max": float(s.max()) if not s.dropna().empty else None,
				"mean": float(round(s.mean(), 2)) if not s.dropna().empty else None,
				"median": float(round(s.median(), 2)) if not s.dropna().empty else None,
				"std": float(round(s.std(), 2)) if not s.dropna().empty else None
			})
			info["sample_values"] = list(pd.Series(s.dropna().unique()).astype(float).round(2).tolist()[:10])
		else:
			top = s.dropna().value_counts().head(10)
			info["top_values"] = top.to_dict()
			info["sample_values"] = list(s.dropna().astype(str).unique()[:10])
	except Exception:
		# Fallback for any odd types
		info["sample_values"] = list(s.dropna().astype(str).unique()[:10])

	return info


def build_column_metadata(df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for col in df.columns:
		summary = summarize_column(df[col])
		rows.append({
			"column": col,
			"dtype": summary.get("dtype"),
			"missing_count": summary.get("missing_count"),
			"missing_pct": summary.get("missing_pct"),
			"unique_count": summary.get("unique_count"),
			"sample_values": json.dumps(summary.get("sample_values"), ensure_ascii=False),
			"top_values": json.dumps(summary.get("top_values", {}), ensure_ascii=False),
			"description": DESCRIPTIONS.get(col, f"No human-friendly description available for column '{col}'.")
		})
	return pd.DataFrame(rows)


@st.cache_data
def load_data(path: str, **kwargs) -> pd.DataFrame:
	return pd.read_csv(path, **kwargs)


st.set_page_config(page_title="Loans - Data Overview", layout="wide")

st.title("Loan Dataset — Overview and Column Descriptions")

st.markdown(
	"""
	This app shows a high-level overview of the cleaned loan dataset (`cleaned_df.csv`).
	Use the sections below to inspect the dataset, review per-column summaries (data type, missing values, unique values, top values),
	edit or add human-friendly column descriptions, and export documentation.
	"""
)

# Dataset-level editable description (prefilled)
if "dataset_description" not in st.session_state:
	st.session_state["dataset_description"] = (
		"Cleaned loan dataset containing borrower, credit, and loan-related fields. "
		"Derived from original loan records and cleaned for analysis (missing columns removed, basic imputation and outlier handling applied)."
	)

st.header("About this dataset")
st.text_area("Dataset description (editable)", value=st.session_state["dataset_description"], height=100, key="dataset_description")

data_path = "cleaned_df.csv"
try:
	df = load_data(data_path, index_col=0)
except FileNotFoundError:
	st.error(f"Could not find {data_path} in the app folder. Make sure the file exists.")
	st.stop()
except Exception as e:
	st.exception(e)
	st.stop()

with st.container():
	c1, c2 = st.columns([2, 1])
	with c1:
		st.header("Dataset preview")
		st.write(f"Rows: {df.shape[0]:,} — Columns: {df.shape[1]}")
		st.dataframe(df.head(100))
	with c2:
		st.header("Quick statistics")
		st.write("**Numeric columns summary**")
		st.dataframe(df.select_dtypes(include=["number"]).describe().T.style.format(precision=2))
		st.write("---")
		st.write("Download a snapshot of the cleaned CSV:")
		st.download_button("Download cleaned_df.csv", data=df.to_csv(index=False).encode("utf-8"), file_name="cleaned_df.csv")

st.header("Per-column summary & description")

meta_df = build_column_metadata(df)

# Prefill session_state descriptions from DESCRIPTIONS so fields are not empty
for col in meta_df["column"]:
	key = f"desc_{col}"
	if key not in st.session_state:
		st.session_state[key] = DESCRIPTIONS.get(col, f"Auto-generated description for {col}.")

# Sidebar filter
filter_term = st.sidebar.text_input("Filter columns (name contains)")
only_with_missing = st.sidebar.checkbox("Show only columns with missing values", value=False)

filtered = meta_df.copy()
if filter_term:
	filtered = filtered[filtered["column"].str.contains(filter_term, case=False, na=False)]
if only_with_missing:
	filtered = filtered[filtered[filtered["missing_count"] > 0].index]

for _, row in filtered.iterrows():
	col = row["column"]
	with st.expander(f"{col} — {row['dtype']} — {row['missing_pct']}% missing"):
		st.write(f"Unique values: {row['unique_count']}")
		st.write("Sample values:")
		try:
			samples = json.loads(row["sample_values"]) if row["sample_values"] else []
		except Exception:
			samples = []
		st.write(samples)
		if row["top_values"] and row["top_values"] != "{}":
			try:
				top_vals = json.loads(row["top_values"])
				st.write("Top values (up to 10):")
				st.write(top_vals)
			except Exception:
				pass

		# If numeric, show a small histogram (use plotly if available, else a simple bar chart)
		if pd.api.types.is_numeric_dtype(df[col]):
			def show_histogram(df_local, column, nbins=40, title=None):
				if _HAS_PLOTLY and px is not None:
					st.plotly_chart(px.histogram(df_local, x=column, nbins=nbins, title=title), use_container_width=True)
				else:
					# fallback: compute histogram and show counts as a bar chart
					arr = df_local[column].dropna().to_numpy()
					if arr.size == 0:
						st.write("No numeric data to display.")
						return
					counts, bin_edges = np.histogram(arr, bins=nbins)
					# create human-readable bin labels
					bins = [f"{round(bin_edges[i],2)} to {round(bin_edges[i+1],2)}" for i in range(len(bin_edges)-1)]
					hist_df = pd.DataFrame({"bin": bins, "count": counts})
					hist_df = hist_df.set_index("bin")
					st.bar_chart(hist_df["count"])

			show_histogram(df, col, nbins=40, title=f"{col} distribution")

	# Editable description
	desc_key = f"desc_{col}"
	# Do NOT assign to st.session_state after widget creation — provide the initial value and let Streamlit manage session_state.
	st.text_area("Column description (editable)", value=st.session_state.get(desc_key, ""), height=120, key=desc_key)

st.markdown("---")
st.header("Export column metadata")
export_df = meta_df.copy()
# Pull user-entered descriptions from session_state where present
export_df["description"] = export_df["column"].apply(lambda c: st.session_state.get(f"desc_{c}", ""))

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download column metadata CSV", data=csv_bytes, file_name="column_metadata.csv")

# Save to disk button
if st.button("Save column metadata to project (column_metadata.csv + dataset_metadata.json)"):
	out_path = os.path.join(os.getcwd(), "column_metadata.csv")
	meta_out = os.path.join(os.getcwd(), "dataset_metadata.json")
	try:
		export_df.to_csv(out_path, index=False)
		# Save dataset-level description and a small metadata object
		dataset_meta = {
			"dataset_description": st.session_state.get("dataset_description", ""),
			"row_count": int(df.shape[0]),
			"column_count": int(df.shape[1])
		}
		with open(meta_out, "w", encoding="utf-8") as f:
			json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

		st.success(f"Saved column metadata to {out_path} and dataset metadata to {meta_out}")
	except Exception as e:
		st.error(f"Failed to save files: {e}")

st.markdown("""
Tips:
- Use the filter box in the sidebar to quickly find columns by name.
- Edit any column description, then download the metadata CSV to save documentation.
""")
