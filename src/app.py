import streamlit as st
import pandas as pd
import tempfile
from data_ingestion import DataIngestion


# Boiler Plate Streamlit code for uploading the csv file or Excel not need to pass path from code

st.set_page_config(page_title="Data Ingestion UI", layout="wide")

st.title("Data Ingestion & Summary Tool")

# Uploading the File
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Save uploaded file to a temp file so DataIngestion can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    st.success(f"File `{uploaded_file.name}` uploaded successfully!")

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(tmp_path)
    else:
        df = pd.read_excel(tmp_path)

    st.dataframe(df.head())

    # Target variable selection
    target = st.selectbox("Select Target Variable (Optional)", ["None"] + list(df.columns))

    if st.button("Generate Summary"):
        if target == "None":
            target = None  # no target selected

        ingestion = DataIngestion(file_path=tmp_path, target_variable=target)
        ingestion.generate_summary()  # prints to console for now
        st.success("✅ Summary generated (check console logs for details)")
else:
    st.info("Please upload a CSV or Excel file to get started.")




