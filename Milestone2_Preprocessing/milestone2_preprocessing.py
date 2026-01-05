"""
Milestone 2: Data Preprocessing and Feature Engineering
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessingPipeline:
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def run_pipeline(self, file_path):
        # Load data
        df = pd.read_excel(file_path) if str(file_path).endswith('.xlsx') else pd.read_csv(file_path)
        print(f"Loaded: {df.shape}")
        
        # Step 1: Handle missing values
        print("\n=== Step 1: Handle Missing Values ===")
        missing = df.isnull().sum()
        print(f"Missing before: {missing.sum()}")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Missing after: {df.isnull().sum().sum()}")
        
        # Step 2: Encode categorical variables
        print("\n=== Step 2: Encode Categorical Variables ===")
        categorical_cols = ['shipment_mode', 'weather_condition', 'region', 'holiday_period', 'carrier_name', 'delayed_reason_code']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        # Step 3: Feature Engineering
        print("\n=== Step 3: Feature Engineering ===")
        if 'supplier_rating' in df.columns and 'previous_on_time_rate' in df.columns:
            df['supplier_reliability_score'] = df['supplier_rating'] * df['previous_on_time_rate']
            print("  Created: supplier_reliability_score")
        
        # Step 4: Normalize numerical features
        print("\n=== Step 4: Normalize Numerical Features ===")
        numerical_cols = ['supplier_rating', 'supplier_lead_time', 'shipping_distance_km', 
                          'order_quantity', 'unit_price', 'total_order_value', 'previous_on_time_rate']
        cols_to_scale = [c for c in numerical_cols if c in df.columns]
        scaled = self.scaler.fit_transform(df[cols_to_scale])
        for i, col in enumerate(cols_to_scale):
            df[f'{col}_scaled'] = scaled[:, i]
        print(f"  Scaled {len(cols_to_scale)} columns")
        
        # Step 5: Train-Test Split
        print("\n=== Step 5: Train-Test Split ===")
        target = 'on_time_delivery'
        exclude = [target, 'order_id', 'supplier_id', 'order_date', 'promised_delivery_date', 
                   'actual_delivery_date'] + categorical_cols
        X = df[[c for c in df.columns if c not in exclude]].select_dtypes(include=[np.number])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(X.columns)}")
        
        # Save outputs
        df.to_csv(self.output_dir / 'processed_data.csv', index=False)
        X_train.to_csv(self.output_dir / 'X_train.csv', index=False)
        X_test.to_csv(self.output_dir / 'X_test.csv', index=False)
        y_train.to_csv(self.output_dir / 'y_train.csv', index=False)
        y_test.to_csv(self.output_dir / 'y_test.csv', index=False)
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.output_dir / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Correlation Heatmap
        print("\n=== Creating Correlation Heatmap ===")
        key_cols = ['supplier_rating', 'supplier_lead_time', 'shipping_distance_km', 
                    'order_quantity', 'total_order_value', 'previous_on_time_rate', 'on_time_delivery']
        plot_cols = [c for c in key_cols if c in df.columns]
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[plot_cols].corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150)
        plt.close()
        
        print(f"\n✓ All outputs saved to: {self.output_dir}")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    base = Path(__file__).parent
    data = base / 'processed_data' / 'SupplyChain_ShipmentSure_Schema.xlsx'
    pipeline = DataPreprocessingPipeline(output_dir=base / 'outputs')
    pipeline.run_pipeline(data)
