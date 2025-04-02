import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load the trained model from the file
model = joblib.load('model.pkl')  # Load the trained Random Forest model
scaler = joblib.load('scaler.pkl')  # Load the scaler if you saved it previously

# Example: new data has 4 features, but we need to make it 11 features to match the training data
# Let's assume the training data had the following features:
# [age, sex, cholesterol, blood_pressure, feature5, feature6, feature7, feature8, feature9, feature10, feature11]

# You need to add the remaining features to make the total number 11. Use appropriate default values for missing features.

# If you only have 4 features, you can add the others as zeros or any other suitable placeholder
new_data = [[79, 0, 1, 90, 200, 1,2,200, 2,1,3]]  # [age, sex, cholesterol, blood_pressure, and 7 other features]


# Step 3: Standardize the new input data using the same scaler used in training
# Ensure the new data has the same features as the training data
new_data_scaled = scaler.transform(new_data)

# Step 4: Make the prediction using the trained model
prediction = model.predict(new_data_scaled)


# Step 5: Output the result
if prediction[0] == 1:
    print("Heart disease detected (1).")
else:
    print("No heart disease detected (0).")
