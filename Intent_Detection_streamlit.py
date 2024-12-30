import streamlit as st
import joblib

model = joblib.load("model.plk")
vectorizer = joblib.load("vectorizer.plk")

st.set_page_config(page_title="Intent Detection by TIFIN", page_icon="🤖", layout="centered")

st.title("🤖 Intent Detection System")
st.write("### Enter your query:")


input = st.text_input("Enter here")

if input:
  input_vector = vectorizer.transform([input])
  prediction = model.predict(input_vector)
  prediction = str(prediction)
  st.write(f"You are looking regarding: {prediction}")

st.markdown("---")
st.markdown("© 2024 TIFIN | Powered by AI")

  