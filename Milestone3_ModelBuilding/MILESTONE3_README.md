# Milestone 3: Model Building and Evaluation

## Overview
This milestone focuses on building, training, and evaluating multiple machine learning models with hyperparameter tuning and comprehensive performance analysis.

## Module Structure

### 1. **model_training.py**
Trains multiple ML models with GridSearchCV for hyperparameter tuning.

#### Key Features:
- **Data Loading & Preparation**
  - Loads preprocessed data from Milestone 2
  - Handles train-test split (80-20)
  - Feature scaling using StandardScaler

- **Three Model Implementations**
  1. **Logistic Regression**
     - Parameters: C, penalty, solver, max_iter
  
  2. **Random Forest**
     - Parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
  
  3. **XGBoost**
     - Parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

- **Hyperparameter Tuning**
  - GridSearchCV with 5-fold cross-validation
  - F1-weighted scoring metric
  - Parallel processing (n_jobs=-1)

#### Usage:
```python
from model_training import ModelTrainer

trainer = ModelTrainer('path/to/processed_data.csv')
if trainer.train_all_models():
    trainer.save_models('trained_models')
```

---

### 2. **model_evaluation.py**
Evaluates trained models using comprehensive metrics.

#### Evaluation Metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among positive predictions
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC-AUC**: Area under the ROC curve

#### Key Methods:
- `load_models_from_disk()`: Load trained models from disk
- `evaluate_model()`: Evaluate a single model
- `evaluate_all_models()`: Evaluate all trained models
- `get_comparison_table()`: Generate comparison DataFrame
- `select_best_model()`: Identify best performing model

#### Usage:
```python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator(X_test, y_test)
evaluator.load_models_from_disk('trained_models')
evaluator.evaluate_all_models()
comparison_table = evaluator.print_comparison_table()
best_model, metrics = evaluator.select_best_model(metric='f1_score')
```

---

### 3. **model_visualizations.py**
Creates comprehensive visualizations of model results.

#### Visualizations Generated:
1. **Confusion Matrix Plots**
   - Individual confusion matrices for each model
   - Combined confusion matrices view

2. **ROC-AUC Curves**
   - Individual ROC curves for each model
   - Combined ROC curves for comparison
   - Supports binary and multi-class classification

3. **Metrics Comparison**
   - Bar charts comparing all metrics across models
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Usage:
```python
from model_visualizations import ModelVisualizer

visualizer = ModelVisualizer(results_dict, y_test, 'visualizations')
visualizer.generate_all_visualizations()
```

---

### 4. **run_pipeline.py**
Complete end-to-end pipeline orchestrating all components.

#### Pipeline Steps:
1. **Data Loading**: Load preprocessed data from Milestone 2
2. **Model Training**: Train all three models with hyperparameter tuning
3. **Model Evaluation**: Evaluate using comprehensive metrics
4. **Visualization**: Generate all plots and comparisons
5. **Reporting**: Create performance summaries

#### Usage:
```bash
python run_pipeline.py
```

This generates:
- Trained models in `trained_models/` directory
- Visualizations in `visualizations/` directory
- Model comparison CSV in `model_comparison.csv`

---

## Configuration

### Data Path
- **Source**: `../Milestone2_Preprocessing/processed_data/SupplyChain_ShipmentSure_Schema.xlsx`
- **Format**: Excel file with preprocessed supply chain data
- **Target Column**: `on_time_delivery` (binary classification: 0 or 1)
- **Features**: 18 input features from supply chain data
- **Samples**: 1000 records
- **Test Size**: 20% (0.2)
- **Random State**: 42 (for reproducibility)

### Model Parameters (GridSearchCV ranges)

**Logistic Regression:**
```
C: [0.001, 0.01, 0.1, 1, 10, 100]
penalty: ['l2']
solver: ['lbfgs', 'liblinear']
max_iter: [200, 500]
```

**Random Forest:**
```
n_estimators: [50, 100, 200]
max_depth: [10, 20, 30, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
max_features: ['sqrt', 'log2']
```

**XGBoost:**
```
n_estimators: [50, 100, 200]
max_depth: [3, 5, 7, 10]
learning_rate: [0.01, 0.05, 0.1]
subsample: [0.7, 0.8, 0.9]
colsample_bytree: [0.7, 0.8, 0.9]
```

---

## Deliverables

### 1. Model Performance Comparison Table
**File**: `model_comparison.csv`
Contains comparison of all models across metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 2. Visualizations

**Confusion Matrices:**
- `visualizations/confusion_matrix_logistic_regression.png`
- `visualizations/confusion_matrix_random_forest.png`
- `visualizations/confusion_matrix_xgboost.png`
- `visualizations/combined_confusion_matrices.png`

**ROC-AUC Curves:**
- `visualizations/roc_curve_logistic_regression.png`
- `visualizations/roc_curve_random_forest.png`
- `visualizations/roc_curve_xgboost.png`
- `visualizations/combined_roc_curves.png`

**Comparisons:**
- `visualizations/metrics_comparison.png`

### 3. Trained Models
**Directory**: `trained_models/`
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `scaler.pkl`

### 4. Final Model Selection
Best model is automatically selected based on F1-Score metric.

---

## Step-by-Step Execution Guide

### Quick Start (Automated):
```bash
# From Milestone3_ModelBuilding directory
python run_pipeline.py
```

### Manual Step-by-Step:

**Step 1: Train Models**
```python
from model_training import ModelTrainer

trainer = ModelTrainer('../Milestone2_Preprocessing/processed_data/processed_data.csv')
trainer.train_all_models()
trainer.save_models('trained_models')
```

**Step 2: Evaluate Models**
```python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator(trainer.X_test, trainer.y_test, trainer.models)
evaluator.evaluate_all_models()
evaluator.print_comparison_table()
```

**Step 3: Visualize Results**
```python
from model_visualizations import ModelVisualizer

visualizer = ModelVisualizer(evaluator.results, trainer.y_test, 'visualizations')
visualizer.generate_all_visualizations()
```

---

## Performance Metrics Explanation

### Accuracy
- Percentage of correct predictions
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Best for balanced datasets

### Precision
- Of positive predictions, how many were correct?
- Formula: TP / (TP + FP)
- Important when false positives are costly

### Recall (Sensitivity)
- Of actual positives, how many did we find?
- Formula: TP / (TP + FN)
- Important when false negatives are costly

### F1-Score
- Harmonic mean of precision and recall
- Formula: 2 * (Precision * Recall) / (Precision + Recall)
- Good balanced metric for imbalanced datasets

### ROC-AUC
- Area Under the Receiver Operating Characteristic Curve
- Measures true positive rate vs false positive rate
- Ranges from 0 to 1 (1.0 is perfect)
- Threshold-independent metric

### Confusion Matrix
- Breakdown of predictions:
  - True Positives (TP): Correct positive predictions
  - True Negatives (TN): Correct negative predictions
  - False Positives (FP): Incorrect positive predictions
  - False Negatives (FN): Incorrect negative predictions

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## Output Structure

```
Milestone3_ModelBuilding/
├── model_training.py
├── model_evaluation.py
├── model_visualizations.py
├── run_pipeline.py
├── MILESTONE3_README.md
├── model_comparison.csv
├── trained_models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
└── visualizations/
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_xgboost.png
    ├── roc_curve_logistic_regression.png
    ├── roc_curve_random_forest.png
    ├── roc_curve_xgboost.png
    ├── combined_confusion_matrices.png
    ├── combined_roc_curves.png
    └── metrics_comparison.png
```

---

## Troubleshooting

### Issue: Data file not found
**Solution**: Ensure Milestone 2 preprocessing is complete and `processed_data.csv` exists in `../Milestone2_Preprocessing/processed_data/`

### Issue: Memory error during GridSearchCV
**Solution**: Reduce the parameter grid sizes or use fewer cross-validation folds

### Issue: ROC-AUC not calculated
**Solution**: Some models may not support `predict_proba()`. Check model compatibility.

### Issue: Visualizations not displaying
**Solution**: Ensure matplotlib backend is configured correctly. Check file paths for write permissions.

---

## Model Selection Criteria

Models are ranked by F1-Score (default), but you can also consider:
- **ROC-AUC**: For threshold-independent evaluation
- **Recall**: If false negatives are more costly
- **Precision**: If false positives are more costly
- **Accuracy**: For overall performance

---

## Next Steps (Milestone 4)

- Model deployment and productionization
- Real-world testing and validation
- Performance monitoring and retraining strategies
- API development for model serving

---

## References

- [Scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [ROC-AUC Explanation](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Confusion Matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)

---

## Author Notes
- All models use stratified train-test split to maintain class distribution
- StandardScaler is used for feature normalization
- GridSearchCV uses 5-fold cross-validation by default
- All results are saved to disk for reproducibility
- Visualizations are saved as high-resolution PNG files (300 DPI)

---

*Last Updated: December 2025*
*Milestone 3 - Model Building and Evaluation Complete*
