"""
Milestone 3: Complete Model Pipeline
Combines training, evaluation, and visualization into one unified script.
"""

import sys
import os
from datetime import datetime

# Import custom modules
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_visualizations import ModelVisualizer


def main():
    """Run the complete model building and evaluation pipeline."""
    
    print("\n" + "="*70)
    print("MILESTONE 3: MODEL BUILDING AND EVALUATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    DATA_PATH = '../Milestone2_Preprocessing/processed_data/SupplyChain_ShipmentSure_Schema.xlsx'
    MODEL_DIR = 'trained_models'
    VIZ_DIR = 'visualizations'
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure Milestone 2 preprocessing is complete.")
        return False
    
    # Step 1: Train Models
    print("\n" + "="*70)
    print("STEP 1: TRAINING MULTIPLE MODELS")
    print("="*70)
    
    trainer = ModelTrainer(DATA_PATH)
    
    if not trainer.train_all_models():
        print("Model training failed!")
        return False
    
    # Save trained models
    trainer.save_models(MODEL_DIR)
    
    # Step 2: Evaluate Models
    print("\n" + "="*70)
    print("STEP 2: EVALUATING TRAINED MODELS")
    print("="*70)
    
    # Extract actual model objects from the trainer.models dictionary
    models_dict = {name: model_info['model'] for name, model_info in trainer.models.items()}
    
    evaluator = ModelEvaluator(
        trainer.X_test,
        trainer.y_test,
        models_dict,
        trainer.scaler
    )
    
    if not evaluator.evaluate_all_models():
        print("Model evaluation failed!")
        return False
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Print detailed reports
    evaluator.print_detailed_report()
    
    # Select best model
    best_model_name, best_metrics = evaluator.select_best_model(metric='f1_score')
    
    # Step 3: Visualizations
    print("\n" + "="*70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = ModelVisualizer(evaluator.results, trainer.y_test, VIZ_DIR)
    visualizer.generate_all_visualizations()
    
    # Step 4: Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nModel Performance Ranking (by F1-Score):")
    print("-" * 50)
    
    # Sort models by F1-Score
    sorted_models = sorted(
        evaluator.results.items(),
        key=lambda x: x[1]['f1_score'],
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name.replace('_', ' ').title()}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ROC-AUC:  {metrics['roc_auc']:.4f}\n")
    
    print("="*70)
    print("\nDeliverables Generated:")
    print("-" * 50)
    print("✓ Model Performance Comparison Table: model_comparison.csv")
    print(f"✓ Confusion Matrices: {VIZ_DIR}/confusion_matrix_*.png")
    print(f"✓ ROC-AUC Curves: {VIZ_DIR}/roc_curve_*.png")
    print(f"✓ Combined Visualizations: {VIZ_DIR}/combined_*.png")
    print(f"✓ Metrics Comparison: {VIZ_DIR}/metrics_comparison.png")
    print(f"✓ Trained Models: {MODEL_DIR}/*.pkl")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
