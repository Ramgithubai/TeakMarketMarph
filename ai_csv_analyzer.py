import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import requests
import json
import warnings
import os
import re
import sys
import asyncio
import tempfile
import io

# Handle environment variables for both local and Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env file if running locally
except ImportError:
    # dotenv not available (e.g., on Streamlit Cloud)
    pass

# Import web scraping module
from modules.web_scraping_module import perform_web_scraping

# Import simplified data explorer
from data_explorer import create_data_explorer

warnings.filterwarnings('ignore')

# Helper function to get environment variables from either .env or Streamlit secrets
def get_env_var(key, default=None):
    \"\"\"Get environment variable from .env file (local) or Streamlit secrets (cloud)\"\"\"
    # First try regular environment variables (from .env or system)
    value = os.getenv(key)
    if value:
        return value

    # Then try Streamlit secrets (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default

# Page configuration
st.set_page_config(
    page_title=\"AI-Powered CSV Data Analyzer\",
    page_icon=\"🤖\",
    layout=\"wide\",
    initial_sidebar_state=\"expanded\"
)

# Main application function
def main():
    \"\"\"Main application function\"\"\"
    
    # App header
    st.markdown('<h1 class=\"main-header\">🚀 TeakMarketMarph - Advanced Business Research Platform</h1>', unsafe_allow_html=True)
    st.markdown(\"**AI-powered market research tool with multi-source data extraction and business contact finding**\")
    
    # File upload section
    st.sidebar.header(\"📁 Data Upload\")
    uploaded_file = st.sidebar.file_uploader(
        \"Upload CSV or Excel file\",
        type=['csv', 'xlsx', 'xls'],
        help=\"Upload your business data for analysis and contact research\"
    )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'identifier_cols' not in st.session_state:
        st.session_state.identifier_cols = []
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner(\"Loading and analyzing your data...\"):
                if uploaded_file.name.endswith('.csv'):
                    df, identifier_cols = smart_csv_loader(uploaded_file)
                else:
                    # For Excel files, let user select sheet if multiple sheets exist
                    excel_file = pd.ExcelFile(uploaded_file)
                    if len(excel_file.sheet_names) > 1:
                        selected_sheet = st.sidebar.selectbox(
                            \"Select Excel Sheet:\",
                            excel_file.sheet_names
                        )
                        df, identifier_cols = smart_excel_loader(uploaded_file, selected_sheet)
                    else:
                        df, identifier_cols = smart_excel_loader(uploaded_file)
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.identifier_cols = identifier_cols
                    st.success(f\"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns\")
        except Exception as e:
            st.error(f\"Error loading file: {str(e)}\")
            st.info(\"Please check your file format and try again\")
    
    # Main application interface
    if st.session_state.df is not None:
        df = st.session_state.df
        identifier_cols = st.session_state.identifier_cols
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            \"📊 Data Overview\",
            \"🔍 Data Explorer\", 
            \"🤖 AI Chat\",
            \"📈 Quick Viz\"
        ])
        
        with tab1:
            create_data_overview(df, st.session_state.get('ai_assistant'), identifier_cols)
        
        with tab2:
            create_data_explorer(df, identifier_cols)
        
        with tab3:
            create_ai_chat_section(df, identifier_cols)
        
        with tab4:
            create_quick_viz(df, identifier_cols)
    
    else:
        # Welcome screen
        st.info(\"👆 Upload a CSV or Excel file to get started with AI-powered business research\")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(\"\"\"
            ### 🔍 Smart Data Analysis
            - Auto-detects HS codes & identifiers
            - AI-powered insights
            - Advanced filtering
            \"\"\")
        
        with col2:
            st.markdown(\"\"\"
            ### 🌐 Business Research
            - Multi-source contact extraction
            - Government data integration
            - JustDial phone numbers
            \"\"\")
        
        with col3:
            st.markdown(\"\"\"
            ### 📊 Export & Visualization
            - Enhanced datasets
            - Interactive charts
            - Download results
            \"\"\")

# Include all the helper functions from the original file here
# (Due to length limitations, I'm including just the essential structure)

if __name__ == \"__main__\":
    main()
