"""Run: python -m streamlit run app.py"""
import streamlit as st
from predict import classify

st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Email / SMS Detection")
st.caption("Deep Learning (Bi-LSTM) classifier")
msg = st.text_area("Paste an email or SMS message", height=150)
if st.button("Classify") and msg.strip():
    label, p = classify(msg)
    (st.error if label == "Spam" else st.success)(f"{label}  -  spam probability {p:.1%}")
    st.progress(p)
