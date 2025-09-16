import streamlit as st
import pandas as pd
import tempfile
import io
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Import all your team's modules.
from data_ingestion import DataIngestion
from data_processing import process_data, to_camel_case
from eda import perform_eda
from feature_engineering import preprocess_features
from model_selection import get_models
from model_training_evaluation import train_and_evaluate_models

# --- 1. THE PIPELINE LOGIC ---
def run_pipeline(file_path: str, target_variable: str):
    all_results = {}
    try:
        ingestion = DataIngestion(file_path=file_path, target_variable=target_variable)
        if ingestion.df is None: return {"error": "Failed to load data."}
        all_results['ingestion'] = {'original_shape': ingestion.df.shape, 'head': ingestion.df.head().to_dict()}

        cleaned_df = process_data(ingestion.df)
        all_results['processing'] = {'cleaned_shape': cleaned_df.shape, 'head': cleaned_df.head().to_dict()}

        if target_variable:
            target_variable = to_camel_case(target_variable)
            if target_variable not in cleaned_df.columns:
                return {"error": f"Target variable '{target_variable}' not found after processing."}

        eda_results = perform_eda(cleaned_df, target_variable=target_variable)
        all_results['eda'] = eda_results
        problem_type = eda_results.get('insights', {}).get('problem_type', 'unsupervised').lower()
        all_results['problem_type'] = problem_type

        models_to_train = get_models(problem_type)
        all_results['model_selection'] = {'models_selected': list(models_to_train.keys())}
        
        if problem_type == 'unsupervised':
            # For unsupervised tasks, scale the numeric features and use the whole dataset.
            processed_df, _, _ = preprocess_features(cleaned_df.select_dtypes(include=np.number))
            train_df, test_df = processed_df, processed_df # Use the same df for train/test
            
            all_results['feature_engineering'] = {
                'processed_shape': processed_df.shape, 
                'head': processed_df.head().to_dict(),
                'message': 'Selected and scaled numeric features for unsupervised task. No train/test split.'
            }
        else:
            # This is the supervised path (Classification/Regression)
            train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
            
            processed_train_df, scaler, encoder = preprocess_features(train_df, target_variable)
            processed_test_df, _, _ = preprocess_features(test_df, target_variable, scaler_obj=scaler, encoder_obj=encoder)
            
            train_df, test_df = processed_train_df, processed_test_df
            
            all_results['feature_engineering'] = {
                'processed_train_shape': train_df.shape, 
                'processed_test_shape': test_df.shape,
                'head': train_df.head().to_dict()
            }
            
        training_results = train_and_evaluate_models(
            train_df, test_df, target_variable, problem_type, models_to_train
        )
            
        all_results['training_evaluation'] = training_results
        return all_results
    except Exception as e:
        return {"error": str(e)}

# --- NEW HELPER FUNCTION TO GENERATE A BEAUTIFUL HTML REPORT ---
def generate_html_report(results: dict) -> str:
    """Converts the results dictionary into a user-friendly, styled HTML report."""
    
    problem_type = results.get('problem_type', 'N/A').capitalize()
    
    # --- CSS Styling ---
    style = """
    <style>
        body { font-family: sans-serif; background-color: #f4f4f9; color: #333; margin: 2em; }
        h1, h2, h3 { color: #2c3e50; }
        .container { background-color: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .section { margin-bottom: 2em; }
        .metric { font-weight: bold; color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin-top: 1em; }
        th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 1em;}
    </style>
    """
    
    # --- HTML Body ---
    html_body = f"<h1>AutoML Pipeline Report</h1>"
    html_body += f"<p><strong>Detected Problem Type:</strong> <span class='metric'>{problem_type}</span></p>"
    
    # --- Ingestion & Processing ---
    html_body += "<div class='section'><h2>Step 1 & 2: Data Ingestion & Processing</h2>"
    ingestion_shape = results.get('ingestion', {}).get('original_shape', (0,0))
    processing_shape = results.get('processing', {}).get('cleaned_shape', (0,0))
    html_body += f"<ul><li><strong>Original Shape:</strong> Rows: <span class='metric'>{ingestion_shape[0]}</span>, Columns: <span class='metric'>{ingestion_shape[1]}</span></li>"
    html_body += f"<li><strong>Shape After Cleaning:</strong> Rows: <span class='metric'>{processing_shape[0]}</span>, Columns: <span class='metric'>{processing_shape[1]}</span></li></ul></div>"
    
    # --- EDA Insights ---
    html_body += "<div class='section'><h2>Step 3: Exploratory Data Analysis</h2><h3>Key Insights</h3><ul>"
    eda_insights = results.get('eda', {}).get('insights', {})
    for key, value in eda_insights.items():
        key_title = key.replace('_', ' ').title()
        html_body += f"<li><strong>{key_title}:</strong> <code>{value}</code></li>"
    html_body += "</ul></div>"
    
    # --- EDA Visualizations ---
    html_body += "<div class='section'><h3>Visualizations</h3>"
    eda_plots = results.get('eda', {}).get('plots', {})
    if not eda_plots:
        html_body += "<p>No visualizations were generated.</p>"
    else:
        for plot_name, plot_base64 in eda_plots.items():
            plot_title = plot_name.replace('_', ' ').title()
            html_body += f"<h4>{plot_title}</h4>"
            html_body += f'<img src="data:image/png;base64,{plot_base64}" alt="{plot_title}">'
    html_body += "</div>"
    
    # --- Model Training & Evaluation ---
    html_body += "<div class='section'><h2>Step 5 & 6: Model Training & Evaluation</h2>"
    eval_results = results.get('training_evaluation', {})
    best_model_name = eval_results.get('best_model_name', 'N/A')
    best_model_score = eval_results.get('best_model_score', 0)
    html_body += f"<p><strong>Best Model Found:</strong> <span class='metric'>{best_model_name}</span> (Primary Score: <span class='metric'>{best_model_score:.4f}</span>)</p>"
    
    html_body += "<h3>All Model Reports</h3>"
    report_df = pd.DataFrame(eval_results.get('all_model_reports', {})).T.reset_index().rename(columns={'index': 'Model'})
    html_body += report_df.to_html(index=False, border=0)
    html_body += "</div>"
    
    return f"<!DOCTYPE html><html><head><title>AutoML Report</title>{style}</head><body><div class='container'>{html_body}</div></body></html>"

# --- 2. THE STREAMLIT APP LOGIC ---
st.set_page_config(page_title="AutoML Pipeline Demo", layout="wide")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    target = None
    run_button = False
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp.seek(0)
                df_for_columns = pd.read_csv(tmp.name, nrows=0) if uploaded_file.name.endswith('.csv') else pd.read_excel(tmp.name, nrows=0)
            target_options = [""] + df_for_columns.columns.tolist()
            target = st.selectbox("Select Target Variable", options=target_options, format_func=lambda x: "None (run as Unsupervised)" if x == "" else x)
            run_button = st.button("🚀 Run AutoML Pipeline")
        except Exception as e:
            st.error(f"Error reading file columns: {e}")

st.title("🤖 AutoML Pipeline Demo")
st.write("A complete demonstration of our team's work, from data ingestion to final model evaluation.")

if not uploaded_file:
    st.info("Please upload a file using the sidebar to begin.")

if 'results' not in st.session_state:
    st.session_state.results = None

if run_button:
    with st.spinner("Running the entire pipeline... this may take a moment."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        st.session_state.results = run_pipeline(file_path=tmp_path, target_variable=target)
        os.remove(tmp_path)

if st.session_state.results:
    results = st.session_state.results
    st.header("Pipeline Results")
    if "error" in results:
        st.error(f"An error occurred in the pipeline: {results['error']}")
    else:
        st.success("Pipeline finished successfully!")
        problem_type_display = results.get('problem_type', 'N/A').capitalize()
        st.info(f"**Detected Problem Type:** {problem_type_display}")

        with st.expander("Step 1 & 2: Data Ingestion & Processing", expanded=True):
            col1, col2 = st.columns(2)
            col1.metric("Original Rows", results.get('ingestion', {}).get('original_shape', (0,0))[0])
            col2.metric("Original Columns", results.get('ingestion', {}).get('original_shape', (0,0))[1])
            st.dataframe(pd.DataFrame(results.get('ingestion', {}).get('head', {})))
            col3, col4 = st.columns(2)
            col3.metric("Rows After Cleaning", results.get('processing', {}).get('cleaned_shape', (0,0))[0])
            col4.metric("Columns After Cleaning", results.get('processing', {}).get('cleaned_shape', (0,0))[1])
            st.dataframe(pd.DataFrame(results.get('processing', {}).get('head', {})))

        with st.expander("Step 3: Exploratory Data Analysis (EDA)", expanded=True):
            st.subheader("Key Insights")
            st.json(results.get('eda', {}).get('insights', {}))
            st.subheader("Visualizations")
            for plot_name, plot_base64 in results.get('eda', {}).get('plots', {}).items():
                st.write(f"**{plot_name.replace('_', ' ').title()}**")
                st.image(f"data:image/png;base64,{plot_base64}")

        with st.expander("Step 4: Feature Engineering", expanded=True):
            st.write(results.get('feature_engineering', {}).get('message', 'Data was split, and transformations were applied.'))
            col5, col6 = st.columns(2)
            if results.get('problem_type') == 'unsupervised':
                shape = results.get('feature_engineering', {}).get('processed_shape', (0,0))
                st.metric("Rows in Processed Dataset", shape[0])
                st.metric("Columns in Processed Dataset", shape[1])
            else:
                shape = results.get('feature_engineering', {}).get('processed_train_shape', (0,0))
                st.metric("Rows in Processed Train Set", shape[0])
                st.metric("Columns in Processed Train Set", shape[1])
            st.dataframe(pd.DataFrame(results.get('feature_engineering', {}).get('head', {})))

        with st.expander("Step 5 & 6: Model Training & Evaluation", expanded=True):
            eval_results = results.get('training_evaluation', {})
            st.success(f"Best Model Found: **{eval_results.get('best_model_name')}** (Score: {eval_results.get('best_model_score', 0):.4f})")
            st.subheader("All Model Reports")
            report_df = pd.DataFrame(eval_results.get('all_model_reports', {})).T.reset_index().rename(columns={'index': 'Model'})
            st.dataframe(report_df)

        # --- UPDATED DOWNLOAD SECTION ---
        st.header("Download Full Report")
        html_report = generate_html_report(results)
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name="automl_pipeline_report.html",
            mime="text/html"
        )

