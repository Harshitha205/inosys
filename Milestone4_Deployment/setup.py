"""
Setup Script for Milestone 4 Deployment
Copies trained models from Milestone 3 to the deployment directory
"""

import pickle
import shutil
from pathlib import Path
import sys


def setup_deployment():
    """Setup the deployment directory with trained models"""
    
    # Define paths
    deployment_dir = Path(__file__).parent
    milestone3_dir = deployment_dir.parent / "Milestone3_ModelBuilding" / "outputs"
    milestone2_dir = deployment_dir.parent / "Milestone2_Preprocessing" / "outputs"
    
    # Create directories if they don't exist
    models_dir = deployment_dir / "trained_models"
    models_dir.mkdir(exist_ok=True)
    
    # Copy trained models
    print("=" * 70)
    print("SETTING UP DEPLOYMENT DIRECTORY")
    print("=" * 70)
    
    model_files = [
        'logistic_regression_model.pkl',
        'random_forest_model.pkl',
        'xgboost_model.pkl',
        'scaler.pkl'
    ]
    
    print("\n📦 Copying trained models...")
    for model_file in model_files:
        source = milestone3_dir / model_file
        dest = models_dir / model_file
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"  ✓ Copied {model_file}")
        else:
            print(f"  ⚠️ Warning: {model_file} not found at {source}")
    
    # Copy preprocessing artifacts
    print("\n🔧 Copying preprocessing artifacts...")
    preprocessing_files = [
        'label_encoders.pkl',
        'X_train.csv',
        'correlation_matrix.csv'
    ]
    
    for prep_file in preprocessing_files:
        source = milestone2_dir / prep_file
        dest = models_dir / prep_file
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"  ✓ Copied {prep_file}")
        else:
            print(f"  ⚠️ Warning: {prep_file} not found at {source}")
    
    # Verify setup
    print("\n✅ Verification:")
    if all((models_dir / f).exists() for f in ['logistic_regression_model.pkl', 'random_forest_model.pkl', 'xgboost_model.pkl']):
        print("  ✓ All models ready for deployment")
    else:
        print("  ⚠️ Some models are missing")
    
    print("\n" + "=" * 70)
    print("Setup complete! Run the app with:")
    print(f"  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        setup_deployment()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)
