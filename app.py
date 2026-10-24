import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Exam Score Prediction",
    page_icon="🎓",
    layout="centered"
)

# Application Title and Description
st.title("🎓 Student Exam Score Prediction System")
st.markdown("This web application predicts a student's final exam score based on academic and lifestyle indicators using a Linear Regression model.")

# Load the Trained Machine Learning Model
@st.cache_resource
def load_prediction_model():
    return joblib.load('student_model.pkl')

try:
    model = load_prediction_model()

    st.subheader("Enter Student Metrics")
    
    # Input Form Fields
    hours_studied = st.number_input(
        "Study Hours (per day)", 
        min_value=0.0, 
        max_value=24.0, 
        value=5.0, 
        step=0.5
    )
    
    sleep_hours = st.number_input(
        "Sleep Duration (hours per night)", 
        min_value=0.0, 
        max_value=24.0, 
        value=8.0, 
        step=0.5
    )
    
    attendance_percent = st.number_input(
        "Attendance Rate (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=80.0, 
        step=1.0
    )
    
    previous_scores = st.number_input(
        "Previous Academic Score (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=70.0, 
        step=1.0
    )

    # Prediction Action Button
    if st.button("Predict Exam Score", type="primary"):
        # Construct Feature DataFrame
        input_features = pd.DataFrame([[
            hours_studied, 
            sleep_hours, 
            attendance_percent, 
            previous_scores
        ]], columns=[
            'hours_studied', 
            'sleep_hours', 
            'attendance_percent', 
            'previous_scores'
        ])
        
        # Perform Inference
        prediction = model.predict(input_features)[0]
        
        # Display Result
        st.success(f"**Predicted Exam Score:** {prediction:.2f} / 100")

except Exception as e:
    st.error(f"Error loading model file: {e}")