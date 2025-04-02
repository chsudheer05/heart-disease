# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import joblib



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Step 2: Load your dataset (replace 'your_dataset.csv' with your actual dataset file path)
df = pd.read_csv("dataset.csv")

# Step 3: Explore the dataset
print(df.head())  # Show the first few rows of the dataset
print(df.describe())  # Show summary statistics
print(df.info())  # Show column types and non-null counts

# Step 4: Data Preprocessing (handling missing values)
df = df.fillna(df.mean())  # Fill missing values with the mean

# Step 5: Feature Selection (split dataset into input and target)
X = df.drop("target", axis=1)  # Features (input variables)
y = df["target"]  # Target (output variable)

# Step 6: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Standardize the data (scale the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Build a Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make Predictions on the Test Data
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Save the trained model to a file
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the model later (if needed)
model = joblib.load('model.pkl')


