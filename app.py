import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Penyakit Ginjal Kronis", layout="wide")

st.title("🏥 Sistem Pendukung Keputusan: Penyakit Ginjal Kronis")
st.markdown("""
Aplikasi ini menggunakan model **XGBoost** untuk memprediksi risiko CKD berdasarkan parameter klinis.
Semua istilah telah disesuaikan ke dalam **Bahasa Indonesia** untuk tenaga medis.
""")

# 1. Fungsi Load Data & Model (Simulasi untuk demo)
@st.cache_resource
def load_resources():
    # Di dunia nyata, kita akan me-load model yang sudah di-save (.pkl)
    # Untuk demo ini, kita asumsikan model dikirim via state atau dilatih cepat
    return None

# Sidebar untuk Input Data
st.sidebar.header("Input Data Medis Pasien")

def user_input_features():
    umur = st.sidebar.slider("Umur", 1, 100, 45)
    tekanan_darah = st.sidebar.slider("Tekanan Darah", 50, 180, 80)
    sg = st.sidebar.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    albumin = st.sidebar.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    gula_urine = st.sidebar.selectbox("Gula Urine", [0, 1, 2, 3, 4, 5])

    hipertensi = st.sidebar.radio("Hipertensi", ["Ya", "Tidak"])
    diabetes = st.sidebar.radio("Diabetes", ["Ya", "Tidak"])
    jantung = st.sidebar.radio("Penyakit Jantung Koroner", ["Ya", "Tidak"])
    edema = st.sidebar.radio("Edema Pedal", ["Ya", "Tidak"])
    anemia = st.sidebar.radio("Anemia", ["Ya", "Tidak"])

    # Numerik lainnya
    gda = st.sidebar.number_input("Gula Darah Acak", 50, 500, 120)
    ureum = st.sidebar.number_input("Ureum Darah", 10, 400, 40)
    sc = st.sidebar.number_input("Serum Kreatinin", 0.1, 15.0, 1.2)
    hemo = st.sidebar.number_input("Hemoglobin", 3.0, 18.0, 12.5)
    sod = st.sidebar.number_input("Natrium", 100, 160, 138)
    pot = st.sidebar.number_input("Kalium", 2.0, 7.0, 4.0)

    data = {
        'Umur': umur, 'Tekanan Darah': tekanan_darah,
        'Hipertensi': 1 if hipertensi == "Ya" else 0,
        'Diabetes': 1 if diabetes == "Ya" else 0,
        'Penyakit Jantung Koroner': 1 if jantung == "Ya" else 0,
        'Specific Gravity': sg, 'Albumin': albumin, 'Gula Urine': gula_urine,
        'Gula Darah Acak': gda, 'Ureum Darah': ureum, 'Serum Kreatinin': sc,
        'Natrium': sod, 'Kalium': pot, 'Hemoglobin': hemo,
        'Edema Pedal': 1 if edema == "Ya" else 0,
        'Anemia': 1 if anemia == "Ya" else 0
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Panel
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ringkasan Input")
    st.write(input_df)

# Load model (disini kita pakai variabel model dari notebook jika di colab,
# namun di streamlit murni harus di-load dari file)
try:
    # Simulasi prediksi (di streamlit server, pastikan model di-export ke model.joblib)
    # model = joblib.load('model_ckd.joblib')
    st.warning("Catatan: Pastikan Anda telah mengekspor model ke file 'model_ckd.joblib' untuk menjalankan aplikasi ini secara mandiri.")
except:
    st.error("Model tidak ditemukan!")
