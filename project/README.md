# CKD Predictor with Explainable AI (XGBoost + SHAP)

Proyek ini adalah aplikasi web untuk memprediksi Penyakit Ginjal Kronis (CKD) menggunakan algoritma XGBoost dan memberikan penjelasan prediksi menggunakan SHAP.

## Fitur Utama
- **Dashboard**: Ringkasan performa model.
- **Prediksi Risiko**: Input parameter klinis pasien untuk mendapatkan diagnosis.
- **Explainable AI**: Visualisasi SHAP untuk transparansi keputusan model.
- **Informasi Medis**: Edukasi mengenai fitur-fitur klinis ginjal.

## Cara Menjalankan Lokal
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Deployment
Siap di-deploy ke [Streamlit Community Cloud](https://streamlit.io/cloud).
