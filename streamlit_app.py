import streamlit as st
import pandas as pd
import requests

# Define the backend API endpoint
BACKEND_URL = 'http://localhost:5000/predict' # Change if Flask app runs elsewhere

# Define the features (new_predictors from your code)
# In a real project, you'd load this from the saved predictors file.
PREDICTORS = [
    'Close_Ratio_2', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 
    'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250', 
    'Close_Ratio_1000', 'Trend_1000'
]

st.title("S&P 500 Direction Predictor 📈")
st.markdown("Enter the calculated market features to predict if the S&P 500 price will go **Up (1)** or **Down/Flat (0)** tomorrow.")

# --- Input Form ---
input_data = {}

st.header("Input Features")
col1, col2 = st.columns(2)

# Create input fields for all 10 predictors
for i, predictor in enumerate(PREDICTORS):
    # Alternate columns for cleaner display
    col = col1 if i % 2 == 0 else col2
    
    # Use st.number_input for float values, with a default for illustration
    default_value = 1.05 if "Ratio" in predictor else 50.0 # Illustrative defaults
    
    input_data[predictor] = col.number_input(
        f"Enter {predictor.replace('_', ' ')}", 
        value=default_value,
        format="%.4f", # Display format for floats
        key=predictor # Unique key for Streamlit
    )

# --- Prediction Button ---
if st.button("Get Prediction"):
    # Convert input data to the format the Flask API expects (JSON)
    payload = input_data

    try:
        # Send POST request to the backend API
        response = requests.post(BACKEND_URL, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            
            # --- Display Results ---
            st.success("✅ Prediction Successful!")
            
            st.subheader(f"Predicted Direction: **{result['message']}** ({int(result['prediction'])})")
            st.metric(
                label="Confidence (Probability of Up)", 
                value=f"{result['probability']:.2f}", 
                help="The model's calculated probability that the price will go up."
            )

        else:
            st.error(f"❌ Backend Error: {response.status_code} - {response.json().get('error', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error(f"🛑 Connection Error: Could not connect to the backend server at `{BACKEND_URL}`. Ensure Flask app is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.markdown("""
---
*Disclaimer: This is a demonstration of an ML model deployment and should not be used for actual financial trading.*
""")