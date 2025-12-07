import streamlit as st
import os
import subprocess
import glob
from pathlib import Path

st.set_page_config(
    page_title="Paleontology Analytics Dashboard",
    page_icon="🦖",
    layout="wide"
)

st.title("🦖 Paleontology Analytics Dashboard")
st.markdown("*Exploring deep time with data science*")

# Paths
ANALYSIS_DIR = "data/analysis"
PROCESSED_DIR = "data/processed"

# Sidebar - Data Info
st.sidebar.header("📊 Dataset Info")
if os.path.exists(f"{PROCESSED_DIR}/merged_occurrences.parquet"):
    import pandas as pd
    df = pd.read_parquet(f"{PROCESSED_DIR}/merged_occurrences.parquet")
    st.sidebar.metric("Total Occurrences", f"{len(df):,}")
    st.sidebar.metric("Unique Genera", f"{df['genus'].nunique():,}")
    st.sidebar.metric("Time Range (Ma)", f"{df['mid_ma'].min():.1f} - {df['mid_ma'].max():.1f}")
else:
    st.sidebar.warning("No processed data found. Run download & normalize first.")

st.sidebar.divider()

# Analysis functions
def run_analysis(analysis_type):
    with st.spinner(f"Running {analysis_type} analysis..."):
        result = subprocess.run(
            ["python3", "-m", "src.cli", "analyze", "--type", analysis_type],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success(f"✅ {analysis_type.upper()} analysis complete!")
            st.rerun()
        else:
            st.error(f"Error: {result.stderr}")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📈 Basic", "🌐 Advanced", "🌍 SOTA", "🤖 ML Extinction"])

# --- BASIC TAB ---
with tab1:
    st.header("Basic Analysis")
    st.markdown("Diversity curves and occurrence maps.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(f"{ANALYSIS_DIR}/diversity_curve.png"):
            st.image(f"{ANALYSIS_DIR}/diversity_curve.png", caption="Diversity Curve (Genera per 5 Ma bin)")
        else:
            st.info("No diversity curve found. Run basic analysis.")
    
    with col2:
        if os.path.exists(f"{ANALYSIS_DIR}/occurrence_map.png"):
            st.image(f"{ANALYSIS_DIR}/occurrence_map.png", caption="Global Occurrence Map")
        else:
            st.info("No occurrence map found. Run basic analysis.")
    
    if st.button("🔄 Rerun Basic Analysis", key="basic"):
        run_analysis("basic")

# --- ADVANCED TAB ---
with tab2:
    st.header("Advanced Analysis")
    st.markdown("Biogeographic networks and subsampled diversity (SQS).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(f"{ANALYSIS_DIR}/biogeographic_network.png"):
            st.image(f"{ANALYSIS_DIR}/biogeographic_network.png", caption="Biogeographic Network")
        else:
            st.info("No network graph found. Run advanced analysis.")
    
    with col2:
        if os.path.exists(f"{ANALYSIS_DIR}/sqs_diversity.png"):
            st.image(f"{ANALYSIS_DIR}/sqs_diversity.png", caption="SQS Diversity Curve")
        else:
            st.info("No SQS curve found. Run advanced analysis.")
    
    if st.button("🔄 Rerun Advanced Analysis", key="advanced"):
        run_analysis("advanced")

# --- SOTA TAB ---
with tab3:
    st.header("SOTA Analysis: The Pulse of Pangea")
    st.markdown("Biogeographic modularity and latitudinal shifts over deep time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(f"{ANALYSIS_DIR}/modularity_over_time.png"):
            st.image(f"{ANALYSIS_DIR}/modularity_over_time.png", caption="Modularity (Provincialism) Over Time")
        else:
            st.info("No modularity plot found. Run SOTA analysis.")
        
        if os.path.exists(f"{ANALYSIS_DIR}/latitudinal_shift.png"):
            st.image(f"{ANALYSIS_DIR}/latitudinal_shift.png", caption="Latitudinal Shift of Diversity")
        else:
            st.info("No latitudinal shift plot found. Run SOTA analysis.")
    
    with col2:
        if os.path.exists(f"{ANALYSIS_DIR}/modularity_vs_diversity.png"):
            st.image(f"{ANALYSIS_DIR}/modularity_vs_diversity.png", caption="Modularity vs. Diversity")
        else:
            st.info("No correlation plot found. Run SOTA analysis.")
    
    if st.button("🔄 Rerun SOTA Analysis", key="sota"):
        run_analysis("sota")

# --- ML TAB ---
with tab4:
    st.header("🤖 ML Extinction Prediction")
    st.markdown("Machine Learning to predict which genera went extinct.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(f"{ANALYSIS_DIR}/extinction_feature_importances.png"):
            st.image(f"{ANALYSIS_DIR}/extinction_feature_importances.png", caption="Feature Importances")
        else:
            st.info("No feature importances plot found. Run ML analysis.")
    
    with col2:
        if os.path.exists(f"{ANALYSIS_DIR}/ml_extinction_summary.txt"):
            st.subheader("📋 Model Summary")
            with open(f"{ANALYSIS_DIR}/ml_extinction_summary.txt", "r") as f:
                summary = f.read()
            st.code(summary)
        else:
            st.info("No ML summary found. Run ML analysis.")
    
    if st.button("🔄 Rerun ML Analysis", key="ml"):
        run_analysis("ml")

# Footer
st.divider()
st.markdown("---")
st.caption("Built with Streamlit • Data from PBDB & Neotoma • 🦕")
