import streamlit as st
import shap
import matplotlib.pyplot as plt

def plot_shap_results(explainer, preprocessor, input_df, feature_names):
    processed_data = preprocessor.transform(input_df)
    shap_values = explainer.shap_values(processed_data)
    
    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[0], feature_names=feature_names, show=False)
    st.pyplot(fig)
    
    st.write("**Interpretasi:** Fitur dengan bar merah/biru teratas adalah faktor yang paling memengaruhi diagnosis pasien ini.")
