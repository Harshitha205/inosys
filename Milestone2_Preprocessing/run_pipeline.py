

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from milestone2_preprocessing import DataPreprocessingPipeline


def run_preprocessing_pipeline():
    """
    Execute the complete preprocessing pipeline.
    
    This function:
    1. Loads the SupplyChain dataset
    2. Handles missing values
    3. Encodes categorical variables
    4. Engineers new features
    5. Normalizes numerical features
    6. Splits data into train/test sets
    7. Creates visualizations
    8. Saves all outputs to 'outputs' folder
    """
    print("="*70)
    print("MILESTONE 2: DATA PREPROCESSING PIPELINE")
    print("SupplyChain ShipmentSure Dataset")
    print("="*70)
    
    # Define paths
    base_path = Path(__file__).parent
    output_dir = base_path / 'outputs'
    
    # Try to find the data file
    possible_paths = [
        base_path / 'processed_data' / 'SupplyChain_ShipmentSure_Schema.xlsx',
        base_path / 'SupplyChain_ShipmentSure_Schema.xlsx',
        base_path / 'sample_data.csv',
        base_path / 'processed_data' / 'processed_data.csv',
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("ERROR: No data file found!")
        print("Please place the SupplyChain_ShipmentSure_Schema.xlsx file in:")
        print(f"  - {base_path / 'processed_data'}")
        print(f"  - or {base_path}")
        return None
    
    print(f"\nData file found: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize pipeline
    pipeline = DataPreprocessingPipeline(output_dir=output_dir)
    
    # Configuration
    config = {
        'encoding_type': 'label',      # 'label' or 'onehot'
        'scaler_type': 'standard',     # 'standard', 'minmax', or 'robust'
        'test_size': 0.2,              # 20% for testing
        'missing_strategy': 'auto'     # 'auto', 'mean', 'median', or 'drop'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run the pipeline
    try:
        X_train, X_test, y_train, y_test, processed_data = pipeline.run_full_pipeline(
            file_path=data_path,
            encoding_type=config['encoding_type'],
            scaler_type=config['scaler_type'],
            test_size=config['test_size'],
            missing_strategy=config['missing_strategy']
        )
        
        # Print summary
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE - SUMMARY")
        print("="*70)
        print(f"\n📊 Dataset Statistics:")
        print(f"   - Original rows: {len(pipeline.raw_data)}")
        print(f"   - Processed columns: {len(processed_data.columns)}")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Features: {len(X_train.columns)}")
        
        print(f"\n📁 Output Files Generated:")
        output_files = sorted(output_dir.glob('*'))
        for file in output_files:
            size = file.stat().st_size / 1024  # KB
            print(f"   - {file.name} ({size:.1f} KB)")
        
        print(f"\n✅ All outputs saved to: {output_dir}")
        print("\n📌 Next Steps:")
        print("   1. Review outputs in the 'outputs' folder")
        print("   2. Proceed to Milestone 3: Model Building")
        print("   3. Use X_train.csv, X_test.csv, y_train.csv, y_test.csv for modeling")
        
        return X_train, X_test, y_train, y_test, processed_data
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def display_output_summary():
    """Display summary of generated outputs."""
    output_dir = Path(__file__).parent / 'outputs'
    
    if not output_dir.exists():
        print("No outputs folder found. Run the pipeline first.")
        return
    
    print("\n" + "="*70)
    print("OUTPUT FILES SUMMARY")
    print("="*70)
    
    # Categorize files
    data_files = []
    report_files = []
    plot_files = []
    model_files = []
    
    for file in output_dir.glob('*'):
        if file.suffix == '.csv':
            data_files.append(file)
        elif file.suffix == '.txt':
            report_files.append(file)
        elif file.suffix == '.png':
            plot_files.append(file)
        elif file.suffix == '.pkl':
            model_files.append(file)
    
    print("\n📊 Data Files:")
    for f in sorted(data_files):
        print(f"   - {f.name}")
    
    print("\n📝 Report Files:")
    for f in sorted(report_files):
        print(f"   - {f.name}")
    
    print("\n📈 Visualization Files:")
    for f in sorted(plot_files):
        print(f"   - {f.name}")
    
    print("\n🔧 Model/Encoder Files:")
    for f in sorted(model_files):
        print(f"   - {f.name}")


if __name__ == "__main__":
    # Run the preprocessing pipeline
    result = run_preprocessing_pipeline()
    
    if result is not None:
        # Display output summary
        display_output_summary()
        
        print("\n" + "="*70)
        print("MILESTONE 2 COMPLETE")
        print("="*70)
