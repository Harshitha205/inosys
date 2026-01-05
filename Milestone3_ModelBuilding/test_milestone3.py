"""
Milestone 3: Unit Tests for Model Training and Evaluation
Tests for the complete model building pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Mock import guard for testing
try:
    from model_training import ModelTrainer
    from model_evaluation import ModelEvaluator
    from model_visualizations import ModelVisualizer
except ImportError:
    print("Warning: Could not import modules. Ensure they're in the Python path.")


class TestDataPreparation(unittest.TestCase):
    """Test data loading and preparation."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Save as CSV
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading(self):
        """Test if data loads correctly."""
        df = pd.read_csv(self.data_path)
        self.assertEqual(df.shape, (200, 11))  # 10 features + 1 target
        self.assertIn('target', df.columns)
        self.assertTrue(np.all(np.isin(df['target'].values, [0, 1])))
    
    def test_data_path_validation(self):
        """Test handling of non-existent data path."""
        invalid_path = '/invalid/path/to/data.csv'
        trainer = ModelTrainer(invalid_path)
        result = trainer.load_and_prepare_data()
        self.assertFalse(result)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_informative=4,
            n_redundant=1,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        df['target'] = y
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(self.data_path)
        self.assertEqual(trainer.test_size, 0.2)
        self.assertEqual(trainer.random_state, 42)
        self.assertIsNone(trainer.X_train)
    
    def test_data_loading_and_preparation(self):
        """Test data loading and splitting."""
        trainer = ModelTrainer(self.data_path)
        result = trainer.load_and_prepare_data()
        
        self.assertTrue(result)
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.X_test)
        self.assertIsNotNone(trainer.y_train)
        self.assertIsNotNone(trainer.y_test)
        
        # Check split ratios
        total_samples = len(trainer.y_train) + len(trainer.y_test)
        self.assertEqual(total_samples, 100)
        self.assertEqual(len(trainer.X_train), 80)
        self.assertEqual(len(trainer.X_test), 20)


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation functionality."""
    
    def setUp(self):
        """Create synthetic test data."""
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 2)
        
        # Create mock predictions
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 2)
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(self.X_test, self.y_test)
        self.assertEqual(len(evaluator.y_test), 20)
        self.assertEqual(len(evaluator.X_test), 20)
        self.assertEqual(len(evaluator.models), 0)
    
    def test_metrics_calculation(self):
        """Test if metrics are calculated correctly."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(np.sum(cm), 20)


class TestVisualizations(unittest.TestCase):
    """Test visualization generation."""
    
    def setUp(self):
        """Create temporary directory for visualizations."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'visualizations')
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test ModelVisualizer initialization."""
        y_test = np.array([0, 1, 0, 1] * 2)
        results = {}
        
        visualizer = ModelVisualizer(results, y_test, self.output_dir)
        self.assertEqual(len(visualizer.results), 0)
        self.assertEqual(len(visualizer.y_test), 8)
        self.assertTrue(os.path.exists(self.output_dir))
    
    def test_output_directory_creation(self):
        """Test if output directory is created."""
        y_test = np.array([0, 1, 0, 1])
        results = {}
        
        output_path = os.path.join(self.temp_dir, 'new_viz')
        visualizer = ModelVisualizer(results, y_test, output_path)
        
        self.assertTrue(os.path.exists(output_path))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Create temporary test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_informative=4,
            n_redundant=1,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        df['target'] = y
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(self.data_path, index=False)
        
        self.model_dir = os.path.join(self.temp_dir, 'trained_models')
        self.viz_dir = os.path.join(self.temp_dir, 'visualizations')
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_data_flow(self):
        """Test data flow through training pipeline."""
        # Load data
        trainer = ModelTrainer(self.data_path)
        result = trainer.load_and_prepare_data()
        
        self.assertTrue(result)
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.y_test)
        
        # Verify data shapes
        self.assertEqual(trainer.X_train.shape[0], 80)
        self.assertEqual(trainer.X_test.shape[0], 20)
        self.assertEqual(len(trainer.y_train), 80)
        self.assertEqual(len(trainer.y_test), 20)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_missing_data_file(self):
        """Test handling of missing data file."""
        trainer = ModelTrainer('/nonexistent/path/data.csv')
        result = trainer.load_and_prepare_data()
        self.assertFalse(result)
    
    def test_empty_results_evaluation(self):
        """Test evaluation with empty results."""
        X_test = np.random.randn(10, 5)
        y_test = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        
        evaluator = ModelEvaluator(X_test, y_test)
        result = evaluator.evaluate_all_models()
        
        # Should fail gracefully
        self.assertFalse(result)


def run_quick_tests():
    """Run quick validation tests."""
    print("\n" + "="*60)
    print("Running Quick Validation Tests")
    print("="*60)
    
    suite = unittest.TestSuite()
    
    # Add basic tests
    suite.addTest(TestDataPreparation('test_data_loading'))
    suite.addTest(TestModelTraining('test_trainer_initialization'))
    suite.addTest(TestModelEvaluation('test_evaluator_initialization'))
    suite.addTest(TestVisualizations('test_visualizer_initialization'))
    suite.addTest(TestErrorHandling('test_missing_data_file'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run quick tests
    success = run_quick_tests()
    
    print("\n" + "="*60)
    if success:
        print("✓ Quick validation tests passed!")
    else:
        print("✗ Some tests failed. Check output above.")
    print("="*60 + "\n")
