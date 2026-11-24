import io
import time
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (
    load_dataset,
    generate_classification_demo,
    generate_regression_demo,
    results_to_dataframe,
    save_sample_datasets
)
from src.preprocessing import (
    detect_task_type,
    detect_feature_types,
    analyze_missing_values,
    analyze_duplicates,
    drop_duplicates,
    analyze_outliers,
    full_preprocess,
    AutoPilotModelPipeline
)
from src.model_trainer import (
    train_all_models,
    get_best_model,
    get_feature_importance,
    get_classification_models,
    get_regression_models,
    tune_model_hyperparameters
)
from src.visualizations import (
    plot_distributions,
    plot_correlation,
    plot_missing,
    plot_categorical,
    plot_outliers,
    plot_model_comparison,
    plot_training_time,
    plot_radar,
    plot_confusion_matrix,
    plot_residuals,
    plot_feature_importance
)

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPilot ML | Enterprise AutoML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### AutoPilot ML\n\nDeveloped by **Param Jaiswal**\n\nEnterprise Automated Machine Learning Platform."
    }
)

# ── Professional SaaS Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container background & spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    /* Navbar / Header */
    .top-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 14px;
        padding: 22px 28px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .header-tagline {
        font-size: 0.95rem;
        color: #93c5fd;
        margin-top: 4px;
        margin-bottom: 0;
        font-weight: 500;
    }

    /* Modern KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #2563eb;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Section container */
    .saas-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Best model banner */
    .trophy-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
    }
    .trophy-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
    }
    .trophy-meta {
        font-size: 0.92rem;
        color: #e0e7ff;
        margin-top: 6px;
    }

    /* Badge tags */
    .badge-primary {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-success {
        background: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Navigation styling in sidebar */
    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ───────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "None"
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "prep" not in st.session_state:
    st.session_state["prep"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "best_name" not in st.session_state:
    st.session_state["best_name"] = None
if "pipeline_obj" not in st.session_state:
    st.session_state["pipeline_obj"] = None

# Auto-generate sample files on disk if not present (safely ignored in read-only containers)
try:
    if not os.path.exists("sample_data/customer_churn_classification.csv"):
        save_sample_datasets()
except Exception:
    pass

# ── Sidebar Navigation & Controls ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size: 1.5rem;">⚡</span>
        <span>AutoPilot ML</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Enterprise Automated Machine Learning")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Dataset Explorer",
            "3. Data Analytics",
            "4. AutoML Training",
            "5. Model Comparison",
            "6. Predictions",
            "7. Model Export"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("📁 Data Source Quick Switch")
    data_option = st.selectbox(
        "Load Dataset",
        ["Select...", "Upload Custom File", "Demo: Customer Churn (Classification)", "Demo: Real Estate (Regression)"],
        index=0
    )

    if data_option == "Upload Custom File":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                loaded_df = load_dataset(uploaded_file)
                st.session_state["df"] = loaded_df
                st.session_state["dataset_name"] = uploaded_file.name
                # Reset previous training states when new dataset is loaded
                st.session_state["target_col"] = loaded_df.columns[-1]
                st.session_state["results"] = None
                st.session_state["prep"] = None
                st.session_state["best_name"] = None
                st.success(f"Loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif data_option == "Demo: Customer Churn (Classification)":
        if st.session_state.get("dataset_name") != "Customer Churn (Demo)":
            demo_df = generate_classification_demo(n=400)
            st.session_state["df"] = demo_df
            st.session_state["dataset_name"] = "Customer Churn (Demo)"
            st.session_state["target_col"] = "churn"
            st.session_state["results"] = None
            st.session_state["prep"] = None
            st.session_state["best_name"] = None
            st.success("Loaded Churn Demo (400 rows)")

    elif data_option == "Demo: Real Estate (Regression)":
        if st.session_state.get("dataset_name") != "Real Estate (Demo)":
            demo_df = generate_regression_demo(n=400)
            st.session_state["df"] = demo_df
            st.session_state["dataset_name"] = "Real Estate (Demo)"
            st.session_state["target_col"] = "price_usd"
            st.session_state["results"] = None
            st.session_state["prep"] = None
            st.session_state["best_name"] = None
            st.success("Loaded Real Estate Demo (400 rows)")

    st.markdown("---")
    # Quick session status
    if st.session_state["df"] is not None:
        st.markdown(f"**Active Dataset**: `{st.session_state['dataset_name']}`")
        st.markdown(f"**Shape**: `{st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols`")
        if st.session_state["best_name"]:
            st.markdown(f"**Trained Model**: 🏆 `{st.session_state['best_name']}`")
    else:
        st.info("No dataset currently active.")

    if st.button("🔄 Reset All State", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About AutoPilot ML", expanded=False):
        st.markdown("""
        **AutoPilot ML**  
        **Developer**: Param Jaiswal  
        **Role**: Machine Learning Engineer  
        **Edition**: Enterprise v2.0
        """)

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div>
        <h1 class="header-title">⚡ AutoPilot ML</h1>
        <p class="header-tagline">Automated Machine Learning & Intelligent Predictive Analytics Platform</p>
    </div>
    <div style="text-align: right;">
        <span class="badge-success">System Status: Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.get("df")

# ═════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW PAGE
# ═════════════════════════════════════════════════════════════════════════════
if nav_selection == "1. Overview":
    st.markdown("### 🌐 Platform Overview")
    st.markdown(
        "Welcome to **AutoPilot ML**, an enterprise automated machine learning platform designed to streamline the "
        "entire data science workflow from data ingestion and exploratory analysis to model training, evaluation, "
        "inference, and deployment."
    )

    # 4 Quick KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Active Dataset</div>
            <div class="kpi-value">""" + ("1" if df is not None else "0") + """</div>
            <div class="kpi-sub">""" + (st.session_state["dataset_name"] if df is not None else "None Loaded") + """</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Dataset Size</div>
            <div class="kpi-value">""" + (f"{df.shape[0]:,}" if df is not None else "0") + """</div>
            <div class="kpi-sub">""" + (f"{df.shape[1]} Features" if df is not None else "Rows") + """</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        n_models = len(st.session_state["results"]) if st.session_state.get("results") else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Trained Models</div>
            <div class="kpi-value">{n_models}</div>
            <div class="kpi-sub">{"Ready for Inference" if n_models > 0 else "Awaiting Training"}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_label = st.session_state.get("best_name") or "None"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Top Performer</div>
            <div class="kpi-value" style="font-size: 1.35rem;">{best_label}</div>
            <div class="kpi-sub">{"Validated" if best_label != "None" else "Pending Run"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Guided Architecture & Workflow Cards
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("#### 🚀 Guided AutoML Workflow")
        workflow_steps = [
            ("1. Dataset Explorer", "Upload CSV or Excel files (.csv, .xlsx, .xls) or load pre-packaged real-world demo datasets. Inspect schema, detect duplicates, and clean missing values."),
            ("2. Data Analytics", "Explore data distributions, interactive correlation matrices, categorical value distributions, and IQR-based anomaly/outlier detection."),
            ("3. AutoML Training", "Select your prediction target. The platform automatically detects regression vs. classification (with manual override), tunes feature scaling, and trains up to 10 algorithms simultaneously."),
            ("4. Model Comparison", "Review the competitive leaderboard, compare performance metrics (R², RMSE, MAE or Accuracy, Precision, Recall, F1), inspect confusion matrices and feature importance."),
            ("5. Predictions", "Run real-time predictions via dynamic feature input forms or upload test datasets for instant batch inference."),
            ("6. Model Export", "Download production-ready serialized inference pipelines (.pkl) that wrap both preprocessing and the trained model.")
        ]
        for title, desc in workflow_steps:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
                <strong style="color: #1e3a8a; font-size: 0.95rem;">{title}</strong>
                <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.88rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### ⚡ Quick Start")
        if df is None:
            st.info("To begin exploring, select a sample dataset below or upload your own file.")
            if st.button("📊 Load Customer Churn (Classification Demo)", use_container_width=True, type="primary"):
                demo_df = generate_classification_demo(n=400)
                st.session_state["df"] = demo_df
                st.session_state["dataset_name"] = "Customer Churn (Demo)"
                st.session_state["target_col"] = "churn"
                st.rerun()

            if st.button("🏠 Load Real Estate (Regression Demo)", use_container_width=True):
                demo_df = generate_regression_demo(n=400)
                st.session_state["df"] = demo_df
                st.session_state["dataset_name"] = "Real Estate (Demo)"
                st.session_state["target_col"] = "price_usd"
                st.rerun()
        else:
            st.success(f"Dataset `{st.session_state['dataset_name']}` is loaded and ready.")
            st.markdown(f"- **Total Rows**: `{df.shape[0]:,}`")
            st.markdown(f"- **Total Columns**: `{df.shape[1]}`")
            st.markdown(f"- **Suggested Target**: `{st.session_state.get('target_col', df.columns[-1])}`")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 Explore Dataset", use_container_width=True):
                    st.session_state["_nav"] = "2. Dataset Explorer"
                    st.rerun()
            with col_b:
                if st.button("🚀 Go to AutoML", use_container_width=True, type="primary"):
                    st.session_state["_nav"] = "4. AutoML Training"
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 👨‍💻 Engineering & Development")
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h4 style="margin: 0; color: #0f172a; font-weight: 700;">Developed by Param Jaiswal</h4>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.9rem;">Lead Developer & Machine Learning Engineer</p>
            </div>
            <span class="badge-primary">AutoPilot ML Enterprise</span>
        </div>
        <p style="margin-top: 12px; color: #475569; font-size: 0.88rem; line-height: 1.5; margin-bottom: 0;">
            AutoPilot ML is engineered to provide end-to-end automated machine learning capabilities: from multi-format dataset ingestion and interactive exploratory data analysis, to parallel model training, comparative diagnostic evaluation, real-time prediction, and standalone pipeline serialization.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "2. Dataset Explorer":
    st.markdown("### 📂 Dataset Explorer & Data Management")
    st.markdown("Upload, inspect, and clean datasets. Supports `.csv`, `.xlsx`, and `.xls` files.")

    # Ingestion Card
    with st.expander("📥 Ingest New Dataset", expanded=(df is None)):
        u_col1, u_col2 = st.columns([1.5, 1])
        with u_col1:
            up_file = st.file_uploader("Upload CSV or Excel spreadsheet", type=["csv", "xlsx", "xls"], key="explorer_uploader")
            if up_file is not None:
                try:
                    df = load_dataset(up_file)
                    st.session_state["df"] = df
                    st.session_state["dataset_name"] = up_file.name
                    st.session_state["target_col"] = df.columns[-1]
                    st.session_state["results"] = None
                    st.session_state["prep"] = None
                    st.success(f"Successfully loaded {up_file.name}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        with u_col2:
            st.markdown("**Sample Datasets:**")
            if st.button("Load Churn Classification Dataset", use_container_width=True):
                st.session_state["df"] = generate_classification_demo(n=400)
                st.session_state["dataset_name"] = "Customer Churn (Demo)"
                st.session_state["target_col"] = "churn"
                st.session_state["results"] = None
                st.session_state["prep"] = None
                st.rerun()
            if st.button("Load Real Estate Regression Dataset", use_container_width=True):
                st.session_state["df"] = generate_regression_demo(n=400)
                st.session_state["dataset_name"] = "Real Estate (Demo)"
                st.session_state["target_col"] = "price_usd"
                st.session_state["results"] = None
                st.session_state["prep"] = None
                st.rerun()

            # Direct in-memory sample download buttons for cloud users
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            churn_csv = generate_classification_demo(n=250).to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Sample Churn CSV", churn_csv, "sample_churn.csv", "text/csv", use_container_width=True)
            estate_csv = generate_regression_demo(n=250).to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Sample Real Estate CSV", estate_csv, "sample_real_estate.csv", "text/csv", use_container_width=True)

    if df is None:
        st.info("👈 Please upload a dataset or select a sample dataset to view the explorer.")
        st.stop()

    # KPI Metrics for Dataset
    m1, m2, m3, m4, m5 = st.columns(5)
    total_nulls = int(df.isnull().sum().sum())
    total_dups = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    m1.metric("Total Records", f"{df.shape[0]:,}")
    m2.metric("Total Features", df.shape[1])
    m3.metric("Numeric Features", len(num_cols))
    m4.metric("Categorical Features", len(cat_cols))
    m5.metric("Missing Values", f"{total_nulls:,}", delta=f"{total_dups} Duplicates", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Data Preview & Cleaning Actions
    st.markdown("#### 📋 Dataset Preview")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=100, value=15, step=5)
    st.dataframe(df.head(preview_rows), use_container_width=True)

    # Duplicate Handling Tool
    st.markdown("#### 🛠️ Data Cleaning & Duplicate Handling")
    dup_info = analyze_duplicates(df)
    c_clean1, c_clean2 = st.columns([1.5, 1])
    with c_clean1:
        if dup_info["duplicate_count"] > 0:
            st.warning(f"Detected **{dup_info['duplicate_count']}** duplicate rows ({dup_info['duplicate_percentage']}% of the dataset).")
            if st.button("🧹 Remove Duplicate Rows Now", type="primary"):
                df_cleaned = drop_duplicates(df)
                st.session_state["df"] = df_cleaned
                st.success(f"Removed {dup_info['duplicate_count']} duplicates! New row count: {df_cleaned.shape[0]}")
                st.rerun()
        else:
            st.success("✅ Zero duplicate rows found in this dataset.")

    # Schema & Column Info
    st.markdown("#### 🧬 Column Schema & Types")
    col_summary = []
    for col in df.columns:
        col_summary.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].notnull().sum()),
            "Null Count": int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
        })
    st.dataframe(pd.DataFrame(col_summary), use_container_width=True)

    # Missing Value Breakdown
    st.markdown("#### 🔍 Missing Values Analysis")
    missing_df = analyze_missing_values(df)
    if missing_df["Missing Count"].sum() > 0:
        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.dataframe(missing_df[missing_df["Missing Count"] > 0], use_container_width=True)
        with col_m2:
            fig_miss = plot_missing(df)
            if fig_miss:
                st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("✅ Dataset is completely clean! No missing values detected.")


# ═════════════════════════════════════════════════════════════════════════════
# 3. DATA ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "3. Data Analytics":
    st.markdown("### 📊 Exploratory Data Analytics (EDA)")
    st.markdown("Comprehensive statistical summaries, feature distributions, correlation matrix, and anomaly detection.")

    if df is None:
        st.info("👈 Please load a dataset first via the sidebar or Dataset Explorer.")
        st.stop()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab_stat, tab_dist, tab_corr, tab_outlier = st.tabs([
        "📈 Statistical Summaries",
        "📊 Distributions",
        "🔥 Correlation Heatmap",
        "🚨 Outliers & Anomalies"
    ])

    with tab_stat:
        st.markdown("#### Numerical Feature Summary Statistics")
        if num_cols:
            desc = df[num_cols].describe().T
            desc["skew"] = df[num_cols].skew()
            desc["median"] = df[num_cols].median()
            st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("No numerical features available.")

        if cat_cols:
            st.markdown("#### Categorical Feature Value Counts")
            selected_cat = st.selectbox("Select Categorical Feature", cat_cols)
            vc = df[selected_cat].value_counts(dropna=False).reset_index()
            vc.columns = [selected_cat, "Count"]
            vc["Percentage (%)"] = np.round((vc["Count"] / len(df)) * 100, 2)
            st.dataframe(vc, use_container_width=True)

    with tab_dist:
        st.markdown("#### Numerical Distributions")
        fig_dist = plot_distributions(df, num_cols)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)

        if cat_cols:
            st.markdown("#### Categorical Distributions")
            fig_cat = plot_categorical(df, cat_cols)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)

    with tab_corr:
        st.markdown("#### Feature Correlation Heatmap")
        fig_corr = plot_correlation(df)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 numerical columns to compute correlations.")

    with tab_outlier:
        st.markdown("#### Interquartile Range (IQR) Outlier Detection")
        outliers_df = analyze_outliers(df, num_cols)
        if not outliers_df.empty:
            st.dataframe(outliers_df, use_container_width=True)
            st.markdown("#### Outlier Box Plots")
            fig_out = plot_outliers(df, num_cols)
            if fig_out:
                st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No numerical features available for outlier analysis.")


# ═════════════════════════════════════════════════════════════════════════════
# 4. AUTOML TRAINING
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "4. AutoML Training":
    st.markdown("### 🚀 Automated Machine Learning Training Engine")
    st.markdown("Configure preprocessing pipelines, select candidate algorithms, and train models in parallel.")

    if df is None:
        st.info("👈 Please load a dataset first via the sidebar or Dataset Explorer.")
        st.stop()

    t_col1, t_col2 = st.columns([1, 1.2])

    with t_col1:
        st.markdown("#### 🎯 Target & Task Configuration")
        default_target_idx = len(df.columns) - 1
        current_target = st.session_state.get("target_col", df.columns[default_target_idx])
        if current_target not in df.columns:
            current_target = df.columns[default_target_idx]
        
        target_col = st.selectbox(
            "Select Target Column (What you want to predict)",
            df.columns.tolist(),
            index=df.columns.tolist().index(current_target)
        )
        st.session_state["target_col"] = target_col

        # Automatic Task Detection
        auto_detected_type = detect_task_type(df, target_col)
        task_mode = st.radio(
            "Task Type",
            ["Auto-Detect", "Manual Override"],
            horizontal=True
        )

        if task_mode == "Auto-Detect":
            task_type = auto_detected_type
            st.info(f"🤖 Auto-detected task type: **{task_type.upper()}** (based on target distribution & data type)")
        else:
            task_type = st.selectbox(
                "Manual Task Selection",
                ["classification", "regression"],
                index=0 if auto_detected_type == "classification" else 1
            )
            st.warning(f"Using manual override: **{task_type.upper()}**")

        st.markdown("#### ⚙️ Feature Preprocessing Options")
        impute_strategy = st.selectbox(
            "Missing Value Imputation",
            ["auto", "median", "mean", "drop"],
            index=0,
            help="Strategy to replace missing values. 'auto' selects median for numeric and mode for categorical."
        )
        scaler_type = st.selectbox(
            "Feature Scaling",
            ["standard", "minmax", "robust", "none"],
            index=0,
            help="Scales features before model fitting. 'standard' is z-score scaling, 'minmax' scales to [0,1]."
        )
        handle_outliers = st.selectbox(
            "Outlier Treatment",
            ["none", "clip"],
            index=0,
            help="'clip' caps numerical features at 1st and 99th percentiles."
        )
        test_size = st.slider("Test Validation Split Ratio", 0.10, 0.40, 0.20, 0.05)

    with t_col2:
        st.markdown("#### 🤖 Model Selection")
        available_models = (
            get_classification_models() if task_type == "classification" else get_regression_models()
        )
        st.caption(f"Found {len(available_models)} algorithms available for {task_type.title()} tasks.")
        
        selected_models = st.multiselect(
            "Select candidate models to evaluate",
            list(available_models.keys()),
            default=list(available_models.keys())
        )

        st.markdown("#### 🧬 Feature Selection")
        candidate_features = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect(
            "Included Features",
            candidate_features,
            default=candidate_features
        )

        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀 Start AutoML Training", type="primary", use_container_width=True)

    if train_btn:
        if not selected_features:
            st.error("Please select at least one feature.")
            st.stop()
        if not selected_models:
            st.error("Please select at least one candidate model.")
            st.stop()

        training_df = df[selected_features + [target_col]].copy()

        with st.status("Running AutoML Pipeline...", expanded=True) as status_box:
            st.write("🔧 Step 1/3: Analyzing schema and applying preprocessing pipeline...")
            prep = full_preprocess(
                training_df,
                target_col=target_col,
                task_type=task_type,
                impute_strategy=impute_strategy,
                scaler_type=scaler_type,
                handle_outliers=handle_outliers,
                test_size=test_size
            )
            st.session_state["prep"] = prep
            if prep.get("sampled_message"):
                st.info(f"⚡ {prep['sampled_message']}")
            st.write(f"✓ Engineered {prep['X_train'].shape[1]} input features from {len(selected_features)} raw columns.")

            st.write(f"⚡ Step 2/3: Training {len(selected_models)} models...")
            progress_bar = st.progress(0.0)

            def update_progress(val, model_name):
                progress_bar.progress(val)

            results = train_all_models(
                prep["X_train"],
                prep["X_test"],
                prep["y_train"],
                prep["y_test"],
                task_type=task_type,
                selected_models=selected_models,
                progress_callback=update_progress
            )
            st.session_state["results"] = results
            progress_bar.progress(1.0)
            st.write("✓ All candidate models evaluated successfully.")

            st.write("🏆 Step 3/3: Selecting best model and building inference pipeline...")
            best_name, best_info = get_best_model(results, task_type)
            st.session_state["best_name"] = best_name

            # Build full serializable inference pipeline
            pipeline_obj = AutoPilotModelPipeline(
                model=best_info["model"],
                preprocessor=prep["preprocessor"],
                task_type=task_type,
                target_col=target_col,
                feature_cols=prep["feature_cols"],
                num_cols=prep["num_cols"],
                cat_cols=prep["cat_cols"],
                target_encoder=prep["target_encoder"],
                original_categories=prep["original_categories"]
            )
            st.session_state["pipeline_obj"] = pipeline_obj

            importances = get_feature_importance(best_info["model"], prep["transformed_feature_names"])
            st.session_state["importances"] = importances

            status_box.update(label=f"✅ AutoML Training Complete! Best Model: {best_name}", state="complete")

        st.success(f"🏆 Best Performing Model: **{best_name}**")
        st.info("Navigate to **Model Comparison** to inspect metrics, or **Predictions** to generate new inferences.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "5. Model Comparison":
    st.markdown("### 🏆 Model Comparison & Evaluation Leaderboard")
    st.markdown("In-depth evaluation, comparative charts, confusion matrices, and feature importance.")

    results = st.session_state.get("results")
    prep = st.session_state.get("prep")
    best_name = st.session_state.get("best_name")

    if not results or not prep:
        st.info("👈 No models trained yet. Please run training from the **AutoML Training** section.")
        st.stop()

    task_type = prep["task_type"]
    best_info = results[best_name]

    # Best Model Banner
    primary_metric_name = "F1 Score" if task_type == "classification" else "R2 Score"
    primary_score = best_info["metrics"].get(primary_metric_name, 0.0)

    st.markdown(f"""
    <div class="trophy-banner">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 class="trophy-title">🏆 Champion Model: {best_name}</h3>
                <div class="trophy-meta">Task: {task_type.upper()} &nbsp;|&nbsp; Primary Metric ({primary_metric_name}): <strong>{primary_score}</strong> &nbsp;|&nbsp; Latency: {best_info['training_time']}s</div>
            </div>
            <div>
                <span class="badge-success" style="font-size: 0.9rem; padding: 6px 14px;">Rank #1 Leader</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Leaderboard Table
    st.markdown("#### 📊 Comprehensive Leaderboard")
    df_results = results_to_dataframe(results)
    st.dataframe(df_results, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparative Charts
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("#### 📊 Metric Comparison")
        fig_comp = plot_model_comparison(results, task_type)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)

    with c_chart2:
        st.markdown("#### ⏱️ Training Latency")
        fig_time = plot_training_time(results)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Task Specific Diagnostic Charts
    if task_type == "classification":
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.markdown(f"#### 🎯 Confusion Matrix ({best_name})")
            cm = best_info.get("extra", {}).get("confusion_matrix")
            classes = prep.get("target_classes") or best_info.get("extra", {}).get("classes")
            fig_cm = plot_confusion_matrix(cm, classes)
            if fig_cm:
                st.plotly_chart(fig_cm, use_container_width=True)

        with c_diag2:
            st.markdown("#### 🕸️ Top Models Radar Chart")
            fig_radar = plot_radar(results, task_type)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.markdown(f"#### 📉 Regression Residuals & Predictions ({best_name})")
        y_test = prep["y_test"]
        y_pred = best_info.get("y_pred")
        fig_res = plot_residuals(y_test, y_pred)
        if fig_res:
            st.plotly_chart(fig_res, use_container_width=True)

    # Feature Importance
    importances = st.session_state.get("importances")
    if importances:
        st.markdown(f"#### 🔍 Key Feature Importances ({best_name})")
        fig_imp = plot_feature_importance(importances, top_n=12)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)

    # Hyperparameter Tuning with Optuna
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("✨ Hyperparameter Optimization (Optuna Engine)", expanded=False):
        st.markdown(f"Fine-tune hyperparameters for champion model **{best_name}** using Bayesian optimization.")
        h_col1, h_col2 = st.columns([1, 2])
        with h_col1:
            trials_count = st.slider("Optimization Trials", min_value=5, max_value=20, value=8, step=1)
            tune_btn = st.button("🚀 Run Optuna Optimization", type="primary")

        if tune_btn:
            with st.spinner(f"Running Optuna Bayesian optimization ({trials_count} trials)..."):
                tuned_m, best_params, best_score = tune_model_hyperparameters(
                    best_name, prep["X_train"], prep["y_train"], task_type, n_trials=trials_count
                )
                st.session_state["pipeline_obj"].model = tuned_m
                st.session_state["results"][best_name]["model"] = tuned_m
                st.success(f"Optimization complete! Cross-validation score: **{best_score:.4f}**")
                with h_col2:
                    st.markdown("**Optimal Parameters:**")
                    st.json(best_params)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "6. Predictions":
    st.markdown("### 🔮 Real-Time & Batch Predictions")
    st.markdown("Run single inference via dynamic form or batch predict by uploading a file.")

    pipeline_obj = st.session_state.get("pipeline_obj")
    prep = st.session_state.get("prep")
    best_name = st.session_state.get("best_name")

    if not pipeline_obj or not prep:
        st.info("👈 Please train a model first in **AutoML Training** before generating predictions.")
        st.stop()

    pred_mode = st.radio("Prediction Mode", ["Single Record Inference", "Batch Inference (File Upload)"], horizontal=True)

    if pred_mode == "Single Record Inference":
        st.markdown(f"#### 📝 Interactive Input Form (Predict with {best_name})")
        st.caption("Fields are pre-populated with median / mode values from the training distribution.")

        feature_cols = prep["feature_cols"]
        num_cols = prep["num_cols"]
        cat_cols = prep["cat_cols"]
        orig_categories = prep["original_categories"]
        X_train_raw = prep["X_train_raw"]

        input_data = {}
        cols = st.columns(3)

        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                if col in cat_cols:
                    options = orig_categories.get(col, [])
                    if not options and col in X_train_raw:
                        options = [str(x) for x in X_train_raw[col].dropna().unique()]
                    input_data[col] = st.selectbox(col, options=options, key=f"pred_{col}")
                else:
                    default_val = float(X_train_raw[col].median()) if col in X_train_raw else 0.0
                    input_data[col] = st.number_input(col, value=default_val, key=f"pred_{col}")

        if st.button("🔮 Generate Prediction", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = pipeline_obj.predict(input_df)[0]

            st.markdown("---")
            p_col1, p_col2 = st.columns([1, 1.2])

            with p_col1:
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 5px solid #2563eb;">
                    <div class="kpi-title">Predicted Value</div>
                    <div class="kpi-value" style="color: #2563eb;">{prediction}</div>
                    <div class="kpi-sub">Target: {prep['target_col']}</div>
                </div>
                """, unsafe_allow_html=True)

            with p_col2:
                if prep["task_type"] == "classification":
                    try:
                        proba = pipeline_obj.predict_proba(input_df)[0]
                        classes = pipeline_obj.target_encoder.classes_ if pipeline_obj.target_encoder else [f"Class {i}" for i in range(len(proba))]
                        proba_df = pd.DataFrame({
                            "Class": [str(c) for c in classes],
                            "Probability": proba
                        }).sort_values("Probability", ascending=False)
                        st.markdown("**Confidence Distribution:**")
                        st.bar_chart(proba_df.set_index("Class"))
                    except Exception:
                        pass

    else:
        st.markdown("#### 📁 Batch Prediction Engine")
        st.caption("Upload a CSV or Excel file containing the required feature columns.")
        test_file = st.file_uploader("Upload Test Dataset", type=["csv", "xlsx", "xls"], key="batch_uploader")

        if test_file is not None:
            try:
                batch_df = load_dataset(test_file)
                st.write(f"Loaded test dataset: {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")

                if st.button("🚀 Run Batch Predictions", type="primary"):
                    with st.spinner("Generating batch predictions..."):
                        preds = pipeline_obj.predict(batch_df)
                        result_df = batch_df.copy()
                        result_df[f"Predicted_{prep['target_col']}"] = preds

                        st.success(f"Successfully generated predictions for all {len(preds):,} records!")
                        st.dataframe(result_df.head(25), use_container_width=True)

                        # CSV Download
                        csv_data = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download Predictions CSV",
                            csv_data,
                            "batch_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error during batch prediction: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. MODEL EXPORT
# ═════════════════════════════════════════════════════════════════════════════
elif nav_selection == "7. Model Export":
    st.markdown("### 📦 Model Export & Production Deployment")
    st.markdown("Download serialized inference pipelines, metrics reports, and integration code snippets.")

    pipeline_obj = st.session_state.get("pipeline_obj")
    prep = st.session_state.get("prep")
    results = st.session_state.get("results")
    best_name = st.session_state.get("best_name")

    if not pipeline_obj or not prep or not results:
        st.info("👈 Please train a model first in **AutoML Training** before exporting.")
        st.stop()

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        st.markdown("#### 💾 Serialized Pipeline Package (.pkl)")
        st.markdown(
            "The download includes the full **`AutoPilotModelPipeline`** wrapping the trained model, "
            "preprocessor, target encoder, and feature schemas. Works standalone in any Python application."
        )

        buffer = io.BytesIO()
        joblib.dump(pipeline_obj, buffer)
        buffer.seek(0)

        clean_model_name = best_name.replace(" ", "_").lower()
        st.download_button(
            label=f"⬇ Download Best Pipeline ({best_name})",
            data=buffer,
            file_name=f"autopilot_{clean_model_name}_pipeline.pkl",
            mime="application/octet-stream",
            type="primary",
            use_container_width=True
        )

    with e_col2:
        st.markdown("#### 📄 Leaderboard Evaluation Report (.csv)")
        st.markdown("Export the complete performance comparison metrics and training times across all candidate models.")

        df_results = results_to_dataframe(results)
        csv_results = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Leaderboard CSV",
            data=csv_results,
            file_name="autopilot_model_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💻 Production Integration Code Snippet")
    st.caption("Copy this snippet to use your exported model in production services, REST APIs, or pipelines:")

    code_snippet = f"""import joblib
import pandas as pd

# 1. Load the serialized pipeline package
pipeline = joblib.load('autopilot_{clean_model_name}_pipeline.pkl')

# 2. Prepare new raw input data (CSV, JSON, or DataFrame)
new_data = pd.DataFrame([
    # Add your raw feature inputs here (no manual scaling or encoding required)
])

# 3. Generate predictions directly
predictions = pipeline.predict(new_data)
print("Predicted:", predictions)

# 4. For classification models, get probabilities:
# probabilities = pipeline.predict_proba(new_data)
"""
    st.code(code_snippet, language="python")

# ── Global Application Footer ──────────────────────────────────────────────────
st.markdown("""
<div style="border-top: 1px solid #e2e8f0; margin-top: 56px; padding-top: 24px; padding-bottom: 24px; text-align: center; color: #64748b; font-size: 0.88rem;">
    <span>⚡ <strong>AutoPilot ML</strong> &nbsp;|&nbsp; Developed by <strong>Param Jaiswal</strong> &nbsp;|&nbsp; Enterprise Automated Machine Learning</span>
</div>
""", unsafe_allow_html=True)
