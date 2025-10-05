import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model and predictors
try:
    with open('sp500_predictor_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('sp500_predictors.pkl', 'rb') as pred_file:
        predictors = pickle.load(pred_file)
except FileNotFoundError:
    print("Error: Model or predictors file not found. Ensure you've run the saving step.")
    model = None
    predictors = []

app = Flask(__name__)

# Define the prediction function using your logic
def make_prediction(input_data_df, model, predictors):
    # The input_data_df must have the same columns as the predictors
    
    # 1. Fit is skipped, as the model is already trained and loaded
    # 2. Predict probability
    preds_proba = model.predict_proba(input_data_df[predictors])[:, 1]
    
    # 3. Apply your custom threshold
    prediction_value = 1 if preds_proba[0] >= 0.6 else 0
    
    return prediction_value, preds_proba[0]

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get data from POST request (e.g., historical market data to calculate features)
    # The data received must be raw S&P 500 data (like 'Close', 'Target')
    # and the feature calculation logic would need to be re-run here.

    # --- SIMPLIFIED DUMMY INPUT FOR DEMO ---
    # In a real system, the frontend would provide the *raw* data needed to calculate 
    # the 10 feature values (Close_Ratio_2, Trend_2, ..., Close_Ratio_1000, Trend_1000).
    # For simplicity, we assume the frontend sends the 10 pre-calculated features.
    
    data = request.json
    
    # Ensure all required features are present
    if not all(p in data for p in predictors):
        return jsonify({'error': f'Missing required features: {predictors}'}), 400

    # Create a DataFrame from the input features
    input_df = pd.DataFrame([data])
    
    # Re-order columns to match the training data
    input_df = input_df[predictors]

    # Make prediction
    pred_val, pred_proba = make_prediction(input_df, model, predictors)
    
    # Return the result
    return jsonify({
        'prediction': pred_val,
        'probability': float(pred_proba),
        'message': 'Up' if pred_val == 1 else 'Down or Flat'
    })

if __name__ == '__main__':
    # You would typically run this with a production server like Gunicorn
    app.run(port=5000, debug=True)