import streamlit as st
import joblib
import os

@st.cache_resource
def load_assets():
    model_path = 'model/model_xgboost_final.pkl'
    prep_path = 'model/preprocessing_pipeline.pkl'
    shap_path = 'model/shap_explainer.pkl'
    feat_path = 'model/feature_order.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Please run training first.")
        return None, None, None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    explainer = joblib.load(shap_path)
    feature_names = joblib.load(feat_path)
    
    return model, preprocessor, explainer, feature_names
