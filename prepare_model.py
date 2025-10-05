import pickle
from sklearn.ensemble import RandomForestClassifier

# Assuming 'model' is your trained RandomForestClassifier from the notebook
# and 'new_predictors' is the list of features you used for the final training.
# Retrain the final model once on the full cleaned dataset for production use
# (or use the model trained during the backtesting, depending on your goal)

# Use the improved model definition
final_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Train on the entire 'sp500' DataFrame after dropna (as used in the last backTest step)
# Note: For real-world use, you'd usually train on a final block of data after feature engineering.
# But for this demo, we'll use the 'new_predictors' list you defined.
# You need to ensure 'sp500' is the DataFrame after the feature creation and dropna steps.
# In a full script, you would perform the feature engineering again here.

# Assume a full re-train for the final production model:
# final_model.fit(sp500[new_predictors], sp500['Target'])

# Save the trained model
with open('sp500_predictor_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

# You should also save the list of predictors (features)
with open('sp500_predictors.pkl', 'wb') as file:
    pickle.dump(new_predictors, file)