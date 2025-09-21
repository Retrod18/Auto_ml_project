# Temporary content for src/main.py

import streamlit as st
import pandas as pd
import cowsay

st.set_page_config(layout="wide")

st.title("Deployment Diagnostic Test 🧪")

try:
    st.header("`pandas` was imported successfully!")
    st.code(f"Pandas version: {pd.__version__}")
    
    st.header("`pycowsay` was imported successfully!")
    st.code(cowsay.get_output_string('cow', 'Moo! The requirements.txt file was read!'))

except ImportError as e:
    st.error(f"A ModuleNotFoundError occurred. This means requirements.txt was NOT installed correctly.")
    st.error(f"Error details: {e}")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
