import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Download IBM HR Dataset
url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
df = pd.read_csv(url)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Select important features
features = [
    'Age', 'MonthlyIncome', 'YearsAtCompany',
    'JobSatisfaction', 'WorkLifeBalance', 'OverTime',
    'EnvironmentSatisfaction', 'YearsSinceLastPromotion',
    'NumCompaniesWorked', 'DistanceFromHome'
]

df = df[features + ['Attrition']]

# Encode categorical columns
le = LabelEncoder()
df['OverTime'] = le.fit_transform(df['OverTime'])      # Yes=1, No=0
df['Attrition'] = le.fit_transform(df['Attrition'])    # Yes=1, No=0

# Split data
X = df[features]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Save model
os.makedirs('app', exist_ok=True)
with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved successfully to app/model.pkl ✅")