import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from utils.loader import load_assets
from utils.predictor import predict_risk
from utils.shap_utils import plot_shap_results

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="CKD Explainable AI",
    page_icon="⚕ ️",
    layout="wide"
)

# CSS Kustom
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation
st.sidebar.title("⚕️ CKD Predictor")
menu = st.sidebar.selectbox("Menu Utama", ["Dashboard", "Prediksi CKD", "Analisis Medis", "Tentang Penelitian"])

# Load Assets
model, preprocessor, explainer, feature_names = load_assets()

if menu == "Dashboard":
    st.title("📊 Dashboard Penelitian")
    st.markdown("### Explainable XGBoost untuk Prediksi Penyakit Ginjal Kronis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi Model", "98.12%")
    col2.metric("ROC AUC", "1.00")
    col3.metric("F1-Score", "1.00")
    col4.metric("Algoritma", "XGBoost")

    st.divider()
    st.subheader("Tujuan Aplikasi")
    st.write("Membantu tenaga medis mendeteksi Chronic Kidney Disease (CKD) secara dini dengan transparansi hasil menggunakan SHAP (Explainable AI).")

elif menu == "Prediksi CKD":
    st.title("📝 Input Data Pasien")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Umur", 1, 100, 45)
            bp = st.number_input("Tekanan Darah (bp)", 50, 200, 80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        with c2:
            al = st.slider("Albumin", 0, 5, 0)
            sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
            hemo = st.number_input("Hemoglobin", 3.0, 18.0, 12.0)
        with c3:
            htn = st.selectbox("Hipertensi", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])
            
        submit = st.form_submit_button("Analisis Risiko")

    if submit:
        # Dummy processing for features not in input to match pipeline
        input_data = pd.DataFrame([[age, bp, sg, al, 0, 120, 40, sc, 138, 4.4, hemo, 40, 8000, 5.0, 
                                    'normal', 'normal', 'notpresent', 'notpresent', htn, dm, 'no', 'good', 'no', ane]], 
                                   columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 
                                            'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])
        
        result = predict_risk(model, preprocessor, input_data)
        
        st.subheader("Hasil Diagnosis")
        if result['label'] == "CKD":
            st.error(f"Hasil: {result['label']} (Probabilitas: {result['prob']:.2%})")
        else:
            st.success(f"Hasil: {result['label']} (Probabilitas: {result['prob']:.2%})")
            
        st.divider()
        st.subheader("🧠 Interpretasi Explainable AI (SHAP)")
        plot_shap_results(explainer, preprocessor, input_data, feature_names)

elif menu == "Analisis Medis":
    st.title("📖 Informasi Klinis")
    with st.expander("Serum Creatinine"):
        st.write("Normal: 0.7-1.3 mg/dL. Kenaikan menandakan penurunan fungsi filtrasi ginjal.")
    with st.expander("Hemoglobin"):
        st.write("Penyakit ginjal sering menyebabkan anemia karena rendahnya hormon eritropoietin.")

elif menu == "Tentang Penelitian":
    st.title("ℹ️ Detail Penelitian")
    st.write("Aplikasi ini merupakan bagian dari skripsi: Explainable XGBoost untuk CKD.")
