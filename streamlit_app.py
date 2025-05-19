import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Eğitimde kullanılan objeleri yükle
with open("xgboost_lda_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("lda_transform.pkl", "rb") as f:
    lda = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("le_dict.pkl", "rb") as f:
    le_dict = pickle.load(f)

st.title("Mental Health Risk Tahmin Sistemi (XGBoost + LDA)")

# Girdi al
def user_input():
    age = st.slider("Yaş", 18, 80, 30)
    gender = st.selectbox("Cinsiyet", le_dict["gender"].classes_)
    employment_status = st.selectbox("İstihdam Durumu", le_dict["employment_status"].classes_)
    work_environment = st.selectbox("Çalışma Ortamı", le_dict["work_environment"].classes_)
    mental_health_history = st.selectbox("Geçmiş Ruhsal Sorun", le_dict["mental_health_history"].classes_)
    seeks_treatment = st.selectbox("Tedavi Arıyor mu?", le_dict["seeks_treatment"].classes_)
    stress_level = st.slider("Stres Seviyesi", 0, 10, 5)
    sleep_hours = st.slider("Uyku Süresi", 0.0, 12.0, 7.0)
    physical_activity_days = st.slider("Fiziksel Aktivite Günü", 0, 7, 3)
    depression_score = st.slider("Depresyon Skoru", 0, 30, 15)
    anxiety_score = st.slider("Anksiyete Skoru", 0, 30, 10)
    social_support_score = st.slider("Sosyal Destek Skoru", 0, 100, 50)
    productivity_score = st.slider("Üretkenlik Skoru", 0.0, 100.0, 60.0)

    data = {
        "age": age,
        "gender": le_dict["gender"].transform([gender])[0],
        "employment_status": le_dict["employment_status"].transform([employment_status])[0],
        "work_environment": le_dict["work_environment"].transform([work_environment])[0],
        "mental_health_history": le_dict["mental_health_history"].transform([mental_health_history])[0],
        "seeks_treatment": le_dict["seeks_treatment"].transform([seeks_treatment])[0],
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "physical_activity_days": physical_activity_days,
        "depression_score": depression_score,
        "anxiety_score": anxiety_score,
        "social_support_score": social_support_score,
        "productivity_score": productivity_score
    }

    return pd.DataFrame([data])

# Kullanıcı girdisi
input_df = user_input()

# Normalize et
input_scaled = scaler.transform(input_df)

# LDA dönüşümü
input_lda = lda.transform(input_scaled)

# Tahmin
prediction = model.predict(input_lda)
prediction_label = label_encoder.inverse_transform(prediction)[0]

st.subheader("Tahmin Edilen Ruhsal Sağlık Riski:")
st.write(f"👉 **{prediction_label}**")
