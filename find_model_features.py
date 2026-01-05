import pickle
import pandas as pd
import numpy as np

# Load model
with open('Milestone3_ModelBuilding/outputs/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load training data
X_train = pd.read_csv('Milestone2_Preprocessing/outputs/X_train.csv')
y_train = pd.read_csv('Milestone2_Preprocessing/outputs/y_train.csv')

print(f"X_train shape: {X_train.shape}")
print(f"Model expects {model.n_features_in_} features")
print(f"\nFirst 12 columns: {list(X_train.columns[:12])}")

# Try with first 12 columns
try:
    X_test = X_train.iloc[:1, :12]
    pred = model.predict(X_test)
    print(f"\nPrediction with first 12 columns: {pred}")
    print("✓ First 12 columns work!")
except Exception as e:
    print(f"✗ Error with first 12 columns: {e}")

# Try other combinations
print(f"\n13th column: {X_train.columns[12]}")
print(f"Columns 7-18: {list(X_train.columns[7:18])}")

# Try columns 7-18 (12 columns total)
try:
    X_test = X_train.iloc[:1, 7:19]
    pred = model.predict(X_test)
    print(f"\nPrediction with columns 7-18: {pred}")
    print("✓ Columns 7-18 work!")
except Exception as e:
    print(f"✗ Error with columns 7-18: {e}")
