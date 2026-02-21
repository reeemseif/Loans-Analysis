# Loans — EDA & Streamlit

Project workspace: `c:\Users\Administrator\Desktop\DS\Projects\Titanic`

# Loans — Exploratory Data Analysis & Interactive Dashboard

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Dataset Description](#-dataset-description)
- [Data Preparation & Feature Engineering](#-data-preparation--feature-engineering)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Key Findings](#-key-findings)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Installation & Usage](#-installation--usage)
- [Requirements](#-requirements)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 📋 Project Overview

This repository contains a comprehensive **exploratory data analysis (EDA)** and an **interactive Streamlit dashboard** for a loan lending dataset. The project aims to uncover key risk factors, understand borrower profiles, and provide actionable insights through intuitive visualizations.

**Key Objectives:**
- Analyze loan default patterns and risk drivers
- Understand borrower characteristics and credit behavior
- Build an interactive dashboard for stakeholders to explore the data

---

## 📁 Project Structure
loan-lending-analysis/
│
├── 📄 README.md # Project documentation
├── 📄 requirements.txt # Dependencies
├── 📄 Home.py # Streamlit main entry point
├── 📄 cleaned_df.csv # Cleaned dataset (analysis-ready)
│
├── 📁 pages/ # Streamlit multi-page directory
│ ├── 📄 Univariate Analysis.py # Distributions and summaries
│ ├── 📄 Multivariate Analysis.py # Relationships and correlations
│ ├── 📄 Borrower Profile.py # Borrower characteristics
│ ├── 📄 Risk Analysis.py # Default patterns & risk factors
│ └── 📄 Loan Performance.py # Loan-level KPIs
│
├── 📁 notebooks/ # Jupyter notebooks
│ ├── 📓 loan-lending.ipynb # Main EDA workflow


---

## 📊 Dataset Description

### 👤 Borrower Demographics
| Column | Description | Type |
|--------|-------------|------|
| `emp_title` | Borrower's job title | Categorical |
| `state` | State of residence | Categorical |
| `homeownership` | Housing status (OWN, RENT, MORTGAGE) | Categorical |
| `annual_income` | Yearly income ($) | Numeric |

### 💳 Credit History
| Column | Description | Type |
|--------|-------------|------|
| `delinq_2y` | Number of delinquencies in last 2 years | Numeric |
| `earliest_credit_line` | Date of first credit account | Date |
| `inquiries_last_12m` | Number of credit inquiries in last year | Numeric |
| `total_credit_lines` | Total number of credit accounts (ever) | Numeric |
| `open_credit_lines` | Number of currently open accounts | Numeric |
| `total_credit_limit` | Total available credit ($) | Numeric |
| `total_credit_utilized` | Total balance owed ($) | Numeric |
| `num_collections_last_12m` | Number of accounts sent to collections | Numeric |
| `tax_liens` | Number of tax liens | Numeric |
| `public_record_bankrupt` | Number of bankruptcies | Numeric |
| `num_cc_carrying_balance` | Number of credit cards with balance | Numeric |

### 🏦 Loan Details
| Column | Description | Type |
|--------|-------------|------|
| `loan_purpose` | Purpose of the loan | Categorical |
| `application_type` | Individual or Joint | Categorical |
| `loan_amount` | Amount borrowed ($) | Numeric |
| `term` | Loan length (36 or 60 months) | Categorical |
| `interest_rate` | Annual interest rate (%) | Numeric |
| `installment` | Monthly payment amount ($) | Numeric |
| `grade` | Risk grade (A = safest, G = riskiest) | Categorical |
| `sub_grade` | Detailed risk grade | Categorical |
| `issue_month` | Month loan was originated | Date |

### 🎯 Loan Performance
| Column | Description | Type |
|--------|-------------|------|
| `loan_status` | Current status (Charged Off, Fully Paid, Current) | Categorical |

---

## 🔧 Data Preparation & Feature Engineering

### Step 1: Data Cleaning
- Removed columns with >40% missing values
- Filled missing categorical values with "Unknown"
- Imputed missing numeric values with median
- Converted `issue_month` to datetime format

### Step 2: Zero Value Analysis
- Identified columns with high zero percentages
- Converted rare events to binary flags:
  ```python
  df['has_tax_lien'] = (df['tax_liens'] > 0).astype(int)
  df['has_bankruptcy'] = (df['public_record_bankrupt'] > 0).astype(int)
  df['has_collections'] = (df['num_collections_last_12m'] > 0).astype(int)
  df['has_delinquency'] = (df['delinq_2y'] > 0).astype(int)

### Step 3: Job Title Categorization
Grouped thousands of unique job titles into meaningful categories:

| Category | Example Titles |
|----------|----------------|
| Management | manager, supervisor, director, president, ceo |
| Healthcare | nurse, rn, doctor, medical |
| Education | teacher, professor, educator |
| Technology | engineer, developer, technician, analyst |
| Transportation | driver, truck driver, delivery |
| Sales | sales, account manager, sales representative |
| Business Owner | owner, proprietor, self-employed |
| Skilled Workers | mechanic, electrician, welder, electrician |
| Administrative | administrative assistant, secretary, clerk |
| Service Jobs | server, customer service, bartender |
| Professional | accountant, attorney, consultant |
| Other | All other titles |

---

### Step 4: Outlier Handling
| Method | Description | Purpose |
|--------|-------------|---------|
| **IQR Detection** | Used Q1 - 1.5*IQR to Q3 + 1.5*IQR | Identify extreme values |
| **Capping** | Capped values at 99th percentile | Improve visualizations |
| **Outlier Flags** | Created binary flags for outliers | Track without removing data |

---

### Step 5: Feature Engineering
| New Feature | Formula | Purpose |
|-------------|---------|---------|
| `credit_utilization` | `total_credit_utilized / total_credit_limit` | Measures credit usage (>30% = risky) |
| `loan_to_income` | `loan_amount / annual_income` | Affordability metric |
| `installment_to_income` | `installment / (annual_income/12)` | Monthly payment burden |
| `credit_age_years` | `current_date - earliest_credit_line` | Credit history length |
| `income_bracket` | Binned `annual_income` | Income segmentation |
| `dti_category` | Binned `debt_to_income` | Risk categorization |

## 🚀 Streamlit Dashboard

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Sidebar Filters** | Grade selection, term filter, sample size control |
| **Key Metrics Cards** | Total Loans, Default Rate, Avg Income, Avg Interest Rate |
| **Multi-Page Navigation** | 5 analysis pages with interactive charts |
| **Real-time Filtering** | All charts update instantly with filter changes |
| **Data Export** | Download filtered data as CSV |

---

### Page Structure

| Page | Description | Key Visualizations |
|------|-------------|-------------------|
| **Univariate Analysis** | Distributions of single variables | Histograms, bar charts, pie charts |
| **Multivariate Analysis** | Relationships between variables | Box plots, scatter plots, correlation heatmaps |
| **Borrower Profile** | Demographics and characteristics | Income brackets, job categories, homeownership |
| **Risk Analysis** | Default patterns and risk factors | Default rates by grade, risk driver analysis, rare events |
| **Loan Performance** | Loan-level KPIs | Loan amounts, term comparisons, purpose analysis |

---

### Technical Implementation

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web application |
| **Visualizations** | Plotly Express | Interactive charts |
| **Data Processing** | Pandas | Data manipulation and filtering |
| **Color Palette** | Custom consistent scheme | Professional look across all charts |
| **Defensive Programming** | Fallback handlers | Graceful handling of missing columns |

---

### Key Features Explained

**Sidebar Controls:**
```python
- Grade filter: Multi-select (A, B, C, D, E, F, G)
- Term filter: 36 months / 60 months
- Sample fraction: 5% to 100% for heavy plots


## 📊 Summary of Key Insights

| Finding | Insight |
|---------|---------|
| **Grade System Validated** | Default rate increases consistently from A (5.2%) to G (31.5%) |
| **Past Delinquencies** | 2.75x higher default risk - strongest predictor |
| **DTI > 40%** | 3.8x higher default risk - critical threshold |
| **Credit Utilization > 30%** | 3x higher default risk - key behavioral metric |
| **Rare Events** | Tax liens (3x risk) and bankruptcy (3.5x risk) affect only 3-5% of borrowers |
| **Verified Income** | 30% lower default rate - verification matters |
| **Most Common Purpose** | Debt consolidation (42% of all loans) |
| **Typical Borrower** | $50-75K income, 2-3 credit cards, 8-12 years credit history |