import streamlit as st
import pandas as pd
import joblib

st.title("📚 Student Exam Scores Prediction App")
st.write("Yeh app trained model file (`student_model.pkl`) ka istemal karti hai.")

# Save ki hui pkl file ko load karna
@st.cache_resource
def load_model():
    return joblib.load('student_model.pkl')

try:
    model = load_model()
    
    st.subheader("Apne ghante darj karein:")
    hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    
    if st.button("Predict Exam Score"):
        # Prediction nikalna
        prediction = model.predict([[hours]])
        st.success(f"Tawaqqa ki jane wali score hai: **{prediction[0]:,.2f}**")

except Exception as e:
    st.error(f"Masla aa gaya hai: {e}")