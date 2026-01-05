"""
Milestone 3: Model Training with Hyperparameter Tuning
"""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ModelTrainer:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        
    def train_all(self):
        # Load data
        df = pd.read_excel(self.data_path) if str(self.data_path).endswith('.xlsx') else pd.read_csv(self.data_path)
        print(f"Loaded: {df.shape}")
        
        # Prepare features and target
        target = 'on_time_delivery'
        drop_cols = ['order_id', 'supplier_id', 'order_date', 'promised_delivery_date', 
                     'actual_delivery_date', 'delayed_reason_code', target]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[target]
        
        # Encode categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train models with GridSearchCV
        models_config = {
            'logistic_regression': (
                LogisticRegression(random_state=42),
                {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
            ),
            'random_forest': (
                RandomForestClassifier(random_state=42, n_jobs=-1),
                {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
            ),
            'xgboost': (
                XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
                {'n_estimators': [100, 200], 'max_depth': [5, 10], 'learning_rate': [0.05, 0.1]}
            )
        }
        
        for name, (model, params) in models_config.items():
            print(f"\n=== Training {name} ===")
            grid = GridSearchCV(model, params, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            self.models[name] = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV F1: {grid.best_score_:.4f}")
        
        self.X_test, self.y_test = X_test, y_test
        return self.models
    
    def save_models(self, output_dir='outputs'):
        Path(output_dir).mkdir(exist_ok=True)
        for name, model in self.models.items():
            with open(f"{output_dir}/{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Models saved to {output_dir}/")


if __name__ == "__main__":
    data = Path(__file__).parent.parent / 'Milestone2_Preprocessing/processed_data/SupplyChain_ShipmentSure_Schema.xlsx'
    trainer = ModelTrainer(data)
    trainer.train_all()
    trainer.save_models()
