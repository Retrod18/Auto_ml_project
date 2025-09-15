import streamlit as st
import pandas as pd
import tempfile
import io

# This is the crucial fix for Python's import system.
# It allows the app to find the modules inside the 'src' folder.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now we can import correctly from our modules
from main import run_pipeline


st.set_page_config(page_title="AutoML Pipeline Demo", layout="wide")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    # This section will only appear after a file is uploaded
    if uploaded_file is not None:
        try:
            # Read the file into a dataframe to get column names
            # To prevent caching issues, create a fresh file-like object
            bytes_data = uploaded_file.getvalue()
            file_like_object = io.BytesIO(bytes_data)
            df_for_columns = pd.read_csv(file_like_object, nrows=0) if uploaded_file.name.endswith('.csv') else pd.read_excel(file_like_object, nrows=0)
            
            # Create options for the selectbox
            target_options = [""] + df_for_columns.columns.tolist()
            
            target = st.selectbox(
                "Select Target Variable",
                options=target_options,
                # This function displays a user-friendly name for the "None" option
                format_func=lambda x: "None (run as Unsupervised)" if x == "" else x,
                help="Choose the column you want to predict. Select 'None' for unsupervised tasks like clustering."
            )

            run_button = st.button("🚀 Run AutoML Pipeline")
        
        except Exception as e:
            st.error(f"Error reading file columns: {e}")
            run_button = False
    else:
        run_button = False

# --- Main Page for a Welcome Message ---
st.title("🤖 AutoML Pipeline Demo")
st.write("A complete demonstration of our team's work, from data ingestion to model selection.")

if not uploaded_file:
    st.info("Please upload a file using the sidebar to begin.")

# --- Main Logic: Run the pipeline when the button is pressed ---
if run_button and uploaded_file is not None:
    with st.spinner("Running the entire pipeline... this may take a moment."):
        # Save the uploaded file to a temporary path to pass to the pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Call the pipeline function from main.py
        results = run_pipeline(file_path=tmp_path, target_variable=target)

        # --- Display Results ---
        st.header("Pipeline Results")

        if "error" in results:
            st.error(f"An error occurred in the pipeline: {results['error']}")
        else:
            st.success("Pipeline finished successfully!")

            # Step 1: Ingestion
            with st.expander("Step 1: Data Ingestion", expanded=True):
                st.write("Initial data was loaded.")
                st.subheader("Data Preview (First 5 Rows)")
                ingestion_head = results.get('ingestion', {}).get('head', {})
                if ingestion_head:
                    st.dataframe(pd.DataFrame.from_dict(ingestion_head))
                
                st.subheader("Original Shape")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", results.get('ingestion', {}).get('original_shape', (0,0))[0])
                with col2:
                    st.metric("Original Columns", results.get('ingestion', {}).get('original_shape', (0,0))[1])


            # Step 2: Processing
            with st.expander("Step 2: Data Processing", expanded=True):
                st.write("Data has been cleaned, with missing values handled and outliers capped.")
                st.subheader("Cleaned Data Preview (First 5 Rows)")
                processing_head = results.get('processing', {}).get('head', {})
                if processing_head:
                    st.dataframe(pd.DataFrame.from_dict(processing_head))

                st.subheader("Shape After Cleaning")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows After Cleaning", results.get('processing', {}).get('cleaned_shape', (0,0))[0])
                with col2:
                    st.metric("Columns After Cleaning", results.get('processing', {}).get('cleaned_shape', (0,0))[1])

            # Step 3: EDA
            with st.expander("Step 3: Exploratory Data Analysis (EDA)", expanded=True):
                st.subheader("Key Insights")
                st.json(results.get('eda', {}).get('insights', {}))

                st.subheader("Visualizations")
                plots = results.get('eda', {}).get('plots', {})
                if not plots:
                    st.write("No plots were generated.")
                for plot_name, plot_base64 in plots.items():
                    st.write(f"**{plot_name.replace('_', ' ').title()}**")
                    st.image(f"data:image/png;base64,{plot_base64}")

            # Step 4: Feature Engineering
            with st.expander("Step 4: Feature Engineering", expanded=True):
                st.write("Numerical features were scaled and categorical features were encoded.")
                st.subheader("Processed Data Preview (First 5 Rows)")
                fe_head = results.get('feature_engineering', {}).get('head', {})
                if fe_head:
                    st.dataframe(pd.DataFrame.from_dict(fe_head))

                st.subheader("Shape After Feature Engineering")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows in Final Dataset", results.get('feature_engineering', {}).get('processed_shape', (0,0))[0])
                with col2:
                    st.metric("Columns in Final Dataset", results.get('feature_engineering', {}).get('processed_shape', (0,0))[1])


            # Step 5: Model Selection
            with st.expander("Step 5: Model Selection", expanded=True):
                problem_type = results.get('model_selection', {}).get('problem_type', 'N/A')
                st.success(f"Pipeline identified the task as: **{problem_type.capitalize()}**")
                st.write("Based on this task, the following models have been selected for training:")
                models = results.get('model_selection', {}).get('models_selected', [])
                for model in models:
                    st.markdown(f"- `{model}`")

        # Clean up the temporary file
        os.remove(tmp_path)

