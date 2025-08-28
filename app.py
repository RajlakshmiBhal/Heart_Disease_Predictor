import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Heart Disease Report", layout="centered")

st.title("🏥 Heart Disease Prediction & Poetic Report")
st.write("Upload your patient data to receive a hospital-style report with a poetic touch.")

uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load model and scaler
    model = joblib.load('heart_rf_model.pkl')
    scaler = joblib.load('heart_scaler.pkl')

    # Preprocess
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    for col in bool_cols:
        df[col] = df[col].astype(int)

    df_encoded = pd.get_dummies(df, columns=cat_cols)
    feature_columns = model.feature_names_in_
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df_encoded)
    preds = model.predict(scaled)

    # Add prediction and poetic summary
    df['Heart_Disease_Prediction'] = preds
    df['Poetic_Summary'] = df['Heart_Disease_Prediction'].apply(lambda x:
        "💔 A heart that whispers warnings in silence." if x == 1 else
        "💖 A rhythm steady, untouched by storm.")

    # Add timestamp
    df['Report_Generated_At'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add doctor’s note
    def doctors_note(pred):
        return ("Patient shows signs of cardiac risk. Immediate consultation recommended."
                if pred == 1 else
                "No immediate cardiac risk detected. Maintain healthy lifestyle and regular checkups.")

    df['Doctor_Note'] = df['Heart_Disease_Prediction'].apply(doctors_note)

    st.subheader("🩺 Full Report")
    st.dataframe(df)

    st.download_button("📥 Download Hospital-Style Report", df.to_csv(index=False), "heart_poetic_report.csv", "text/csv")
