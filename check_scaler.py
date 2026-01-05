import pickle
import pandas as pd

# Load the scaler
with open('Milestone2_Preprocessing/outputs/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Check what features it knows about
if hasattr(scaler, 'feature_names_in_'):
    print("Scaler feature names:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"{i+1}. {name}")
else:
    print("Scaler has no feature_names_in_ attribute")

# Also check the n_features_in_
if hasattr(scaler, 'n_features_in_'):
    print(f"\nNumber of features: {scaler.n_features_in_}")

# Try to transform sample data
X_train = pd.read_csv('Milestone2_Preprocessing/outputs/X_train.csv')
print(f"\nX_train shape: {X_train.shape}")
print(f"X_train columns: {list(X_train.columns)}")
