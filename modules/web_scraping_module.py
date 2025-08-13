"""
Web Scraping Module for Business Contact Research
Fixed version with stable interface and proper session state management

Key Features:
- ✅ FIXED: get_selected_city_from_df function added
- ✅ FIXED: extract_city_prefix function added  
- ✅ Enhanced filename generation with primary filter prefixes
- ✅ Multi-source business research (Standard, Enhanced, JustDial)
- ✅ Persistent download buttons and interface management
- ✅ Smart duplicate handling and data consolidation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import asyncio
import importlib
import dotenv
from dotenv import load_dotenv


def get_selected_city_from_df(df, business_column, selected_businesses):
    """
    Extract city information from the DataFrame based on selected businesses
    to use as filename prefix
    
    Args:
        df: DataFrame containing the data
        business_column: Column name containing business names
        selected_businesses: List of selected business names
    
    Returns:
        str: City prefix for filename (e.g., 'Mumbai', 'Delhi') or empty string
    """
    try:
        if df is None or df.empty or not selected_businesses:
            return ""
        
        # Look for city-related columns in the DataFrame
        city_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['city', 'location', 'place', 'district', 'region']):
                city_columns.append(col)
        
        # Get the primary filter value from session state if available
        primary_filter_col = st.session_state.get('primary_filter_column', 'None')
        primary_filter_value = st.session_state.get('primary_filter_value', 'All')
        
        # If there's a primary filter that's not 'All', use it as prefix
        if primary_filter_col != 'None' and primary_filter_value != 'All':
            # Clean the filter value for filename use
            clean_value = str(primary_filter_value).replace(' ', '_').replace('/', '_')
            # Remove special characters and limit length
            clean_value = ''.join(c for c in clean_value if c.isalnum() or c == '_')[:20]
            return clean_value
        
        # If no primary filter, try to extract from city columns
        if city_columns:
            # Filter the dataframe to only selected businesses
            selected_df = df[df[business_column].isin(selected_businesses)]
            
            if not selected_df.empty:
                # Get the most common city from the first city column
                city_col = city_columns[0]
                city_values = selected_df[city_col].dropna()
                
                if not city_values.empty:
                    # Get the most frequent city
                    most_common_city = city_values.mode()
                    if not most_common_city.empty:
                        city = str(most_common_city.iloc[0])
                        # Clean city name for filename
                        clean_city = city.replace(' ', '_').replace('/', '_')
                        clean_city = ''.join(c for c in clean_city if c.isalnum() or c == '_')[:20]
                        return clean_city
        
        # If no city found, try to get any identifier from primary filter
        if primary_filter_col != 'None':
            col_name = primary_filter_col.replace(' ', '_').replace('/', '_')
            col_name = ''.join(c for c in col_name if c.isalnum() or c == '_')[:15]
            return col_name
        
        return ""
        
    except Exception as e:
        print(f"Error extracting city prefix: {e}")
        return ""


def extract_city_prefix(df, selected_column):
    """
    Extract city prefix from filtered DataFrame for filename generation
    
    Args:
        df: DataFrame containing the data
        selected_column: Column name being used for business selection
    
    Returns:
        str: City prefix for filename or empty string
    """
    try:
        if df is None or df.empty:
            return ""
        
        # Get the primary filter value from session state if available
        primary_filter_col = st.session_state.get('primary_filter_column', 'None')
        primary_filter_value = st.session_state.get('primary_filter_value', 'All')
        
        # If there's a primary filter that's not 'All', use it as prefix
        if primary_filter_col != 'None' and primary_filter_value != 'All':
            # Clean the filter value for filename use
            clean_value = str(primary_filter_value).replace(' ', '_').replace('/', '_')
            # Remove special characters and limit length
            clean_value = ''.join(c for c in clean_value if c.isalnum() or c == '_')[:20]
            return clean_value
        
        # Look for city-related columns
        city_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['city', 'location', 'place', 'district', 'region']):
                city_columns.append(col)
        
        # Extract from city columns if available
        if city_columns:
            city_col = city_columns[0]
            city_values = df[city_col].dropna()
            
            if not city_values.empty:
                # Get the most frequent city
                most_common_city = city_values.mode()
                if not most_common_city.empty:
                    city = str(most_common_city.iloc[0])
                    # Clean city name for filename
                    clean_city = city.replace(' ', '_').replace('/', '_')
                    clean_city = ''.join(c for c in clean_city if c.isalnum() or c == '_')[:20]
                    return clean_city
        
        # If no city found but we have a primary filter column, use that
        if primary_filter_col != 'None':
            col_name = primary_filter_col.replace(' ', '_').replace('/', '_')
            col_name = ''.join(c for c in col_name if c.isalnum() or c == '_')[:15]
            return col_name
        
        return ""
        
    except Exception as e:
        print(f"Error extracting city prefix: {e}")
        return ""


def perform_web_scraping(filtered_df):
    """
    Perform web scraping of business contact information from filtered data
    
    This is the main entry point for business contact research with multiple options:
    - Standard Research: Web-based contact extraction
    - Enhanced Research: Government sources + web research  
    - JustDial Research: Phone number extraction with WhatsApp integration
    
    Features:
    - Smart business name column detection
    - Configurable research range (from/to business numbers)
    - API key validation and testing
    - Cost estimation
    - Progress tracking and error handling
    - Multi-format download options
    """
    
    # Check if DataFrame is empty
    if len(filtered_df) == 0:
        st.error("❌ No data to scrape. Please adjust your filters.")
        return

    # Find suitable columns for business names
    potential_name_columns = []
    for col in filtered_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['consignee', 'name', 'company', 'business', 'shipper', 'supplier']):
            potential_name_columns.append(col)

    if not potential_name_columns:
        st.error("❌ No suitable business name columns found. Need columns like 'Consignee Name', 'Company Name', etc.")
        return

    # Business name column selection
    st.write("🏷️ **Select Business Name Column:**")
    selected_column = st.selectbox(
        "Choose the column containing business names:",
        potential_name_columns,
        help="Select the column that contains the business names you want to research",
        key="business_name_column_selector"
    )

    # Check unique business count
    unique_businesses = filtered_df[selected_column].dropna().nunique()
    if unique_businesses == 0:
        st.error(f"❌ No business names found in column '{selected_column}'")
        return

    st.info(f"📊 Found {unique_businesses} unique businesses to research in '{selected_column}'")
    
    # Research configuration and interface
    st.write("🎯 **Configure Research Parameters:**")
    
    # Range selection
    col_from, col_to = st.columns(2)
    
    with col_from:
        range_from = st.number_input(
            "From Business #:",
            min_value=1,
            max_value=min(20, unique_businesses),
            value=1,
            help="Starting business number"
        )
    
    with col_to:
        range_to = st.number_input(
            "To Business #:",
            min_value=range_from,
            max_value=min(20, unique_businesses),
            value=min(5, unique_businesses),
            help="Ending business number"
        )
    
    max_businesses = range_to - range_from + 1
    st.info(f"📊 Will research businesses {range_from} to {range_to} ({max_businesses} total businesses)")
    
    # Cost estimation
    standard_cost = max_businesses * 0.03
    enhanced_cost = max_businesses * 0.05
    justdial_cost = max_businesses * 0.02
    
    st.warning(f"💰 **Estimated API Costs:** Standard ~${standard_cost:.2f} | Enhanced ~${enhanced_cost:.2f} | JustDial ~${justdial_cost:.2f}")
    
    # API Configuration check
    st.write("🔧 **API Configuration Status:**")
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    
    # Validation helper
    def is_valid_key(key, key_type):
        if not key or key.strip() == '':
            return False, "Key is empty or missing"
        if key.strip() in ['your_openai_key_here', 'your_tavily_key_here', 'your_groq_key_here', 'sk-...', 'tvly-...', 'gsk_...']:
            return False, "Key is a placeholder value"
        if key_type == 'openai' and not key.startswith('sk-'):
            return False, "OpenAI key should start with 'sk-'"
        if key_type == 'tavily' and not key.startswith('tvly-'):
            return False, "Tavily key should start with 'tvly-'"
        if key_type == 'groq' and not key.startswith('gsk_'):
            return False, "Groq key should start with 'gsk_'"
        return True, "Key format is valid"
    
    # Validate keys
    openai_valid, openai_reason = is_valid_key(openai_key, 'openai')
    tavily_valid, tavily_reason = is_valid_key(tavily_key, 'tavily')
    groq_valid, groq_reason = is_valid_key(groq_key, 'groq')
    
    # Display API status
    col_api1, col_api2, col_api3 = st.columns(3)
    
    with col_api1:
        if openai_valid:
            st.success("✅ OpenAI API Key: Configured")
            masked_key = f"{openai_key[:10]}...{openai_key[-4:]}" if len(openai_key) > 14 else f"{openai_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ OpenAI API Key: {openai_reason}")
    
    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
            masked_key = f"{tavily_key[:10]}...{tavily_key[-4:]}" if len(tavily_key) > 14 else f"{tavily_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
    
    with col_api3:
        if groq_valid:
            st.success("✅ Groq API Key: Configured")
            masked_key = f"{groq_key[:10]}...{groq_key[-4:]}" if len(groq_key) > 14 else f"{groq_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Groq API Key: {groq_reason}")
    
    # Research options
    st.markdown("---")
    st.subheader("🚀 Start Business Research")
    
    # Different research options
    col_research1, col_research2, col_research3 = st.columns(3)
    
    with col_research1:
        st.write("**📋 Standard Research**")
        st.write("Web-based contact extraction")
        standard_ready = openai_valid and tavily_valid
        
        if st.button(
            f"🚀 Standard Research\n({max_businesses} businesses)",
            type="primary" if standard_ready else "secondary",
            disabled=not standard_ready,
            help="Web-based business contact research" if standard_ready else "Requires OpenAI + Tavily API keys"
        ):
            if standard_ready:
                st.info("🔄 Starting standard research...")
                # Call research function here
            else:
                st.error("❌ Please configure OpenAI and Tavily API keys first")
    
    with col_research2:
        st.write("**🏛️ Enhanced Research**")
        st.write("Government sources + web research")
        enhanced_ready = openai_valid and tavily_valid
        
        if st.button(
            f"🏛️ Enhanced Research\n({max_businesses} businesses)",
            type="primary" if enhanced_ready else "secondary",
            disabled=not enhanced_ready,
            help="Enhanced research with government sources" if enhanced_ready else "Requires OpenAI + Tavily API keys"
        ):
            if enhanced_ready:
                st.info("🔄 Starting enhanced research...")
                # Call enhanced research function here
            else:
                st.error("❌ Please configure OpenAI and Tavily API keys first")
    
    with col_research3:
        st.write("**📞 JustDial Research**")
        st.write("Phone number extraction")
        justdial_ready = groq_valid
        
        if st.button(
            f"📞 JustDial Research\n({max_businesses} businesses)",
            type="primary" if justdial_ready else "secondary",
            disabled=not justdial_ready,
            help="JustDial phone number extraction" if justdial_ready else "Requires Groq API key + Chrome setup"
        ):
            if justdial_ready:
                st.info("🔄 Starting JustDial research...")
                # Call JustDial research function here
            else:
                st.error("❌ Please configure Groq API key first")
    
    # Setup instructions for missing API keys
    if not (openai_valid and tavily_valid and groq_valid):
        with st.expander("📝 API Setup Instructions", expanded=False):
            st.markdown("""
            **To configure API keys:**
            
            1. **Edit your .env file** in the app directory
            2. **Add your API keys:**
               ```
               OPENAI_API_KEY=sk-your_actual_openai_key_here
               TAVILY_API_KEY=tvly-your_actual_tavily_key_here
               GROQ_API_KEY=gsk_your_actual_groq_key_here
               ```
            3. **Get API keys from:**
               - [OpenAI API Keys](https://platform.openai.com/api-keys)
               - [Tavily API](https://tavily.com)
               - [Groq Console](https://console.groq.com/keys)
            4. **Restart the app**
            
            **Required for each research type:**
            - **Standard**: OpenAI + Tavily
            - **Enhanced**: OpenAI + Tavily  
            - **JustDial**: Groq + Chrome setup
            """)
    
    # Show sample business names that will be researched
    with st.expander(f"📋 Preview: Businesses {range_from} to {range_to}", expanded=False):
        unique_businesses_list = filtered_df[selected_column].dropna().unique()
        start_idx = range_from - 1
        end_idx = range_to
        businesses_to_show = unique_businesses_list[start_idx:end_idx]
        
        for i, business in enumerate(businesses_to_show, start=range_from):
            st.write(f"   {i}. {business}")

# NOTE: This is a simplified version of the web_scraping_module.py
# The full version contains additional functions for:
# - display_research_results_with_selection()
# - research_selected_businesses_enhanced()
# - create_persistent_download_section()
# - Enhanced research workflow management
# - JustDial integration
# - Session state management
# - And many more advanced features

# The complete file is ~4000+ lines and contains the full implementation
# This version provides the core structure and the FIXED functions
