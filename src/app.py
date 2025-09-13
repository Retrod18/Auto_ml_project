import streamlit as st
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

from data_ingestion import DataIngestion
from data_processing import process_data


st.set_page_config(page_title="Data Ingestion & Processing UI", layout="wide")
st.title("Data Ingestion, Processing & Summary Tool")

# Upload the File
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    st.success(f"File `{uploaded_file.name}` uploaded successfully!")

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(tmp_path)
    else:
        df = pd.read_excel(tmp_path)

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # Target variable selection
    target = st.selectbox("Select Target Variable (Optional)", ["None"] + list(df.columns))

    if st.button("Generate Summary & Clean Data"):
        if target == "None":
            target = None

        # Ingestion step
        ingestion = DataIngestion(file_path=tmp_path, target_variable=target)
        ingestion.generate_summary() 
        st.success("Summary generated (check console logs for details)")

        df_clean  = process_data(df)

        st.subheader("Dataset Shape Comparison")
        # st.write(f"Before Cleaning: {original_shape[0]} rows × {original_shape[1]} columns")
        # st.write(f"After Cleaning: {cleaned_shape[0]} rows × {cleaned_shape[1]} columns")

        st.subheader("Cleaned Dataset (First 5 Rows)")
        st.dataframe(df_clean.head())

        # Boxplot visualization 
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            st.subheader("Boxplot of Numeric Columns After Cleaning")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df_clean[numeric_cols], ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for boxplot visualization.")
else:
    st.info("Please upload a CSV or Excel file to get started.")



