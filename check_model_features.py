import pickle
import pandas as pd

# Load a model to check its input features
with open('Milestone3_ModelBuilding/outputs/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Check model properties
if hasattr(model, 'n_features_in_'):
    print(f"Model expects {model.n_features_in_} features")

# Load X_train to understand the data
X_train = pd.read_csv('Milestone2_Preprocessing/outputs/X_train.csv')
print(f"\nX_train shape: {X_train.shape}")
print(f"X_train columns: {list(X_train.columns)}")

# Check if there's info in the preprocessing outputs
try:
    with open('Milestone2_Preprocessing/outputs/column_list.txt', 'r') as f:
        print("\nColumn list from preprocessing:")
        print(f.read())
except:
    pass

# Check if we can find info about model training
try:
    info_file = 'Milestone3_ModelBuilding/outputs/model_comparison.csv'
    df = pd.read_csv(info_file)
    print("\nModel comparison info:")
    print(df)
except:
    pass
