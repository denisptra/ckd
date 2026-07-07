import numpy as np

def predict_risk(model, preprocessor, input_df):
    processed_data = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_data)[0][1]
    label = "CKD" if prob > 0.5 else "Non-CKD"
    return {'label': label, 'prob': prob}
