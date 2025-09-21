import streamlit as st
import pandas as pd
import tempfile
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

def run_pipeline(file_path: str, target_variable: str):
    all_results = {}
    try:
        if not target_variable:
            target_variable = None

        ingestion = DataIngestion(file_path=file_path, target_variable=target_variable)
        if ingestion.df is None: return {"error": "Failed to load data."}
        
        cleaned_df = process_data(ingestion.df)
        all_results['processing'] = {'cleaned_shape': cleaned_df.shape, 'head': cleaned_df.head().to_dict('records')}

        if target_variable:
            target_variable_camel = to_camel_case(target_variable)
            if target_variable_camel not in cleaned_df.columns:
                return {"error": f"Target variable '{target_variable}' not found after processing."}
            target_variable = target_variable_camel
        
        eda_results = perform_eda(cleaned_df, target_variable=target_variable)
        all_results['eda'] = eda_results
        problem_type = eda_results.get('insights', {}).get('problem_type', 'unsupervised').lower()
        all_results['problem_type'] = problem_type

        models_to_train = get_models(problem_type)
        
        if problem_type == 'unsupervised':
            # This is the key fix: We pass `target_variable` (which is None) to the function.
            processed_df, _, _ = preprocess_features(cleaned_df, target_variable)
            all_results['feature_engineering'] = {'processed_shape': processed_df.shape, 'head': processed_df.head().to_dict('records')}
            training_results = train_and_evaluate_models(processed_df, processed_df, None, 'clustering', models_to_train)
        else:
            # Supervised path
            train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
            processed_train_df, scaler, encoder = preprocess_features(train_df, target_variable)
            processed_test_df, _, _ = preprocess_features(test_df, target_variable, scaler_obj=scaler, encoder_obj=encoder)
            all_results['feature_engineering'] = {'processed_train_shape': processed_train_df.shape, 'processed_test_shape': processed_test_df.shape, 'head': processed_train_df.head().to_dict('records')}
            training_results = train_and_evaluate_models(processed_train_df, processed_test_df, target_variable, problem_type, models_to_train)
            
        all_results['training_evaluation'] = training_results
        return all_results
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

# (Keep the rest of your Streamlit UI and HTML report generation code as it was, it's correct)
# For completeness, the UI code is included here.

def generate_html_report(results: dict) -> str:
    problem_type = results.get('problem_type', 'N/A').capitalize()
    style = """<style> body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f9fafb; color: #1f2937; } .container { max-width: 1000px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); } h1, h2, h3, h4 { color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; } h1 { text-align: center; border: none; font-size: 2.5em; margin-bottom: 5px; } .section { margin-bottom: 30px; } table { width: 100%; border-collapse: collapse; margin-top: 15px; } th, td { text-align: left; padding: 10px 12px; border: 1px solid #d1d5db; } th { background-color: #3b82f6; color: white; } tr:nth-child(even) { background-color: #f9fafb; } img { max-width: 90%; height: auto; border-radius: 8px; margin: 20px auto; display: block; box-shadow: 0 2px 8px rgba(0,0,0,0.1); } .dataframe { overflow-x: auto; } .subtitle { text-align: center; color: #6b7280; margin-top: 0; margin-bottom: 30px; font-size: 1.1em; } </style>"""
    html = f"<h1>AutoML Pipeline Report</h1><p class='subtitle'><strong>Detected Problem Type:</strong> {problem_type}</p>"
    html += "<h2>Step 1 & 2: Data Ingestion & Processing</h2>"
    processing_res = results.get('processing', {})
    html += f"<p><b>Shape After Cleaning:</b> {processing_res.get('cleaned_shape')}</p>"
    html += "<h4>Data Preview (Cleaned)</h4><div class='dataframe'>"
    html += pd.DataFrame(processing_res.get('head', {})).to_html(classes='styled-table') + "</div>"
    html += "<h2>Step 3: Exploratory Data Analysis</h2>"
    insights = results.get('eda', {}).get('insights', {})
    if 'statistical_summary' in insights:
        html += "<h4>Statistical Summary</h4><div class='dataframe'>" + pd.DataFrame(insights['statistical_summary']).to_html(classes='styled-table') + "</div>"
    html += "<h3>Visualizations</h3>"
    for name, b64_string in results.get('eda', {}).get('plots', {}).items():
        html += f"<h4>{name.replace('_', ' ').title()}</h4><img src='data:image/png;base64,{b64_string}'>"
    html += "<h2>Step 4: Feature Engineering</h2>"
    fe_res = results.get('feature_engineering', {})
    if 'processed_train_shape' in fe_res:
         html += f"<p><b>Processed Train Set Shape:</b> {fe_res.get('processed_train_shape')}</p>"
    else:
         html += f"<p><b>Processed Data Shape:</b> {fe_res.get('processed_shape')}</p>"
    html += "<h2>Step 5 & 6: Model Training & Evaluation</h2>"
    eval_results = results.get('training_evaluation', {})
    html += f"<p><strong>Best Model Found:</strong> {eval_results.get('best_model_name', 'N/A')} (Score: {eval_results.get('best_model_score', 0):.4f})</p>"
    html += "<h3>All Model Reports</h3><div class='dataframe'>"
    report_df = pd.DataFrame(eval_results.get('all_model_reports', {})).T.reset_index().rename(columns={'index': 'Model'})
    html += report_df.to_html(index=False, classes='styled-table') + "</div>"
    return f"<!DOCTYPE html><html><head><title>AutoML Report</title>{style}</head><body><div class='container'>{html}</div></body></html>"

st.set_page_config(page_title="AutoML Pipeline Demo", layout="wide")
if 'results' not in st.session_state: st.session_state.results = None
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    target = None; run_button = False
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_cols = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
            else:
                df_cols = pd.read_excel(uploaded_file, nrows=0).columns.tolist()
            uploaded_file.seek(0)
            target = st.selectbox("Select Target Variable", options=[""] + df_cols, format_func=lambda x: "None (Unsupervised)" if x == "" else x)
            run_button = st.button("🚀 Run AutoML Pipeline")
        except Exception as e: st.error(f"Error reading file columns: {e}")
st.title("🤖 AutoML Pipeline Demo")
st.write("A complete demonstration of our team's work, from data ingestion to final model evaluation.")
if not uploaded_file: st.info("Please upload a file to begin.")
if run_button:
    with st.spinner("Running the entire pipeline... This may take a moment..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
        st.session_state.results = run_pipeline(file_path=tmp_path, target_variable=target)
        os.remove(tmp_path)
if st.session_state.results:
    results = st.session_state.results
    st.header("Pipeline Results")
    if "error" in results:
        st.error(f"An error occurred: {results['error']}")
        if 'traceback' in results: st.code(results['traceback'])
    else:
        st.success("Pipeline finished successfully!")
        with st.expander("Step 1 & 2: Ingestion & Processing", expanded=True):
            st.dataframe(pd.DataFrame(results.get('processing', {}).get('head', {})))
        with st.expander("Step 3: Exploratory Data Analysis (EDA)", expanded=True):
            insights = results.get('eda', {}).get('insights', {})
            st.subheader("Visualizations")
            for name, b64 in results.get('eda', {}).get('plots', {}).items():
                st.write(f"**{name.replace('_', ' ').title()}**"); st.image(f"data:image/png;base64,{b64}")
        with st.expander("Step 5 & 6: Model Training & Evaluation", expanded=True):
            eval_results = results.get('training_evaluation', {})
            st.success(f"Best Model Found: **{eval_results.get('best_model_name')}** (Score: {eval_results.get('best_model_score', 0):.4f})")
            report_df = pd.DataFrame(eval_results.get('all_model_reports', {})).T.reset_index().rename(columns={'index': 'Model'})
            st.dataframe(report_df.style.format(precision=4))
        st.header("Download Full Report")
        html_report = generate_html_report(results)
        st.download_button(label="📥 Download HTML Report", data=html_report, file_name="automl_pipeline_report.html", mime="text/html")