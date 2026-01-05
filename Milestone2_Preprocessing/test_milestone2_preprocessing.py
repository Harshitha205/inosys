"""
UNIT TESTS: Data Preprocessing Pipeline
========================================

Comprehensive test suite for the preprocessing pipeline.
Tests all major functions and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from milestone2_preprocessing import DataPreprocessingPipeline


class TestDataPreprocessingPipeline(unittest.TestCase):
    """Test cases for DataPreprocessingPipeline class"""
    
    @classmethod
    def setUpClass(cls):
        """Create sample test data for all tests"""
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Create sample data
        cls.sample_data = pd.DataFrame({
            'Customer_ID': [1, 2, 3, 4, 5],
            'Product_weight': [100, 200, np.nan, 400, 500],
            'Distance': [1000, 2000, 3000, 4000, 5000],
            'Total_cost': [5000, 10000, 15000, 20000, 25000],
            'Delivery_Time': [5, 10, 15, np.nan, 25],
            'Mode_of_Shipment': ['Flight', 'Ship', 'Road', 'Flight', 'Ship'],
            'Product_importance': ['Low', 'Medium', 'High', 'Low', 'Medium'],
            'Gender': ['M', 'F', 'M', 'F', 'M'],
            'Reached_on_Time_Y_N': [1, 0, 1, 1, 0]
        })
        
        # Save sample data to CSV
        cls.data_path = os.path.join(cls.test_dir, 'test_data.csv')
        cls.sample_data.to_csv(cls.data_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Initialize pipeline for each test"""
        self.pipeline = DataPreprocessingPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNone(self.pipeline.raw_data)
        self.assertIsNone(self.pipeline.processed_data)
        self.assertIsNone(self.pipeline.scaler)
        self.assertEqual(self.pipeline.config['test_size'], 0.2)
        self.assertEqual(self.pipeline.config['random_state'], 42)
    
    def test_load_data_csv(self):
        """Test loading data from CSV file"""
        self.pipeline.load_data(self.data_path)
        
        self.assertIsNotNone(self.pipeline.raw_data)
        self.assertEqual(self.pipeline.raw_data.shape[0], 5)
        self.assertIn('Product_weight', self.pipeline.raw_data.columns)
    
    def test_load_data_invalid_path(self):
        """Test error handling for invalid file path"""
        with self.assertRaises(FileNotFoundError):
            self.pipeline.load_data('nonexistent_file.csv')
    
    def test_handle_missing_values(self):
        """Test missing value imputation"""
        self.pipeline.load_data(self.data_path)
        
        # Check missing values before
        missing_before = self.pipeline.raw_data.isnull().sum().sum()
        self.assertGreater(missing_before, 0)
        
        # Handle missing values
        self.pipeline.step1_handle_missing_values()
        
        # Check missing values after
        missing_after = self.pipeline.raw_data.isnull().sum().sum()
        self.assertEqual(missing_after, 0)
    
    def test_encode_categorical_variables(self):
        """Test categorical encoding"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        
        # Get categorical columns before encoding
        categorical_before = self.pipeline.raw_data.select_dtypes(
            include=['object']
        ).shape[1]
        
        # Encode categorical variables
        self.pipeline.step2_encode_categorical_variables()
        
        # Check encoders were created
        self.assertGreater(len(self.pipeline.encoders), 0)
        
        # Check categorical values were encoded to numeric
        for col in ['Mode_of_Shipment', 'Product_importance', 'Gender']:
            if col in self.pipeline.raw_data.columns:
                self.assertTrue(
                    np.issubdtype(self.pipeline.raw_data[col].dtype, np.number)
                )
    
    def test_engineer_features(self):
        """Test feature engineering"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        
        # Count columns before
        cols_before = self.pipeline.raw_data.shape[1]
        
        # Engineer features
        self.pipeline.step3_engineer_features()
        
        # Count columns after
        cols_after = self.pipeline.raw_data.shape[1]
        
        # Should have more columns after feature engineering
        self.assertGreater(cols_after, cols_before)
        
        # Check if engineered features exist
        engineered_features = [
            'cost_per_unit', 'cost_per_km',
            'lead_time_efficiency', 'order_size_category'
        ]
        
        for feature in engineered_features:
            if feature in self.pipeline.raw_data.columns:
                # Check that engineered feature has values
                self.assertGreater(
                    self.pipeline.raw_data[feature].notna().sum(), 0
                )
    
    def test_normalize_numerical_features(self):
        """Test numerical feature normalization"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        self.pipeline.step3_engineer_features()
        
        # Get numerical columns
        numerical_cols = self.pipeline.raw_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # Get statistics before normalization
        before_mean = self.pipeline.raw_data[numerical_cols[0]].mean()
        before_std = self.pipeline.raw_data[numerical_cols[0]].std()
        
        # Normalize
        self.pipeline.step4_normalize_numerical_features()
        
        # Check scaler was fitted
        self.assertIsNotNone(self.pipeline.scaler)
        
        # Check that normalized data has mean close to 0
        after_mean = abs(self.pipeline.raw_data[numerical_cols[0]].mean())
        self.assertLess(after_mean, 1e-10)  # Should be very close to 0
    
    def test_split_train_test(self):
        """Test train-test splitting"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        self.pipeline.step3_engineer_features()
        self.pipeline.step4_normalize_numerical_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.pipeline.step5_split_train_test()
        
        # Check sizes
        self.assertEqual(len(X_train) + len(X_test), len(self.pipeline.raw_data))
        
        # Check test size ratio
        test_ratio = len(X_test) / (len(X_train) + len(X_test))
        self.assertAlmostEqual(test_ratio, 0.2, places=1)
        
        # Check X and y sizes match
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
    
    def test_feature_correlation_analysis(self):
        """Test feature correlation analysis"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        self.pipeline.step3_engineer_features()
        self.pipeline.step4_normalize_numerical_features()
        
        # Analyze correlations
        corr_matrix = self.pipeline.analyze_feature_correlation()
        
        # Check correlation matrix
        self.assertIsNotNone(corr_matrix)
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Check that diagonal elements are 1 (correlation with self)
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix),
            np.ones(corr_matrix.shape[0])
        )
    
    def test_configuration_custom(self):
        """Test custom configuration"""
        custom_config = {
            'test_size': 0.3,
            'random_state': 123
        }
        
        pipeline = DataPreprocessingPipeline(config=custom_config)
        
        self.assertEqual(pipeline.config['test_size'], 0.3)
        self.assertEqual(pipeline.config['random_state'], 123)
    
    def test_data_summary(self):
        """Test getting data summary"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        self.pipeline.step3_engineer_features()
        self.pipeline.step4_normalize_numerical_features()
        self.pipeline.step5_split_train_test()
        
        summary = self.pipeline.get_data_summary()
        
        self.assertIn('raw_shape', summary)
        self.assertIn('processed_shape', summary)
        self.assertIn('missing_values', summary)
        self.assertIn('categorical_encoders', summary)
        self.assertIn('feature_count', summary)
    
    def test_save_and_load_pipeline_objects(self):
        """Test saving and loading pipeline objects"""
        self.pipeline.load_data(self.data_path)
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        self.pipeline.step3_engineer_features()
        self.pipeline.step4_normalize_numerical_features()
        
        # Save
        scaler_path = os.path.join(self.test_dir, 'test_scaler.pkl')
        encoders_path = os.path.join(self.test_dir, 'test_encoders.pkl')
        
        self.pipeline.save_pipeline_objects(scaler_path, encoders_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(scaler_path))
        self.assertTrue(os.path.exists(encoders_path))
        
        # Load in new pipeline
        new_pipeline = DataPreprocessingPipeline()
        new_pipeline.load_pipeline_objects(scaler_path, encoders_path)
        
        # Check objects loaded
        self.assertIsNotNone(new_pipeline.scaler)
        self.assertGreater(len(new_pipeline.encoders), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.pipeline = DataPreprocessingPipeline()
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        self.pipeline.raw_data = empty_df
        
        # Should handle gracefully
        try:
            self.pipeline.step1_handle_missing_values()
        except Exception as e:
            # Expected for empty data
            pass
    
    def test_all_missing_values_single_column(self):
        """Test handling of column with all missing values"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.nan, np.nan, np.nan]
        })
        self.pipeline.raw_data = df
        
        # Should handle all-missing column
        self.pipeline.step1_handle_missing_values()
    
    def test_single_row_dataframe(self):
        """Test handling of single-row dataframe"""
        df = pd.DataFrame({
            'col1': [1],
            'col2': [2],
            'target': [0]
        })
        self.pipeline.raw_data = df
        
        # Should handle single row
        try:
            self.pipeline.step4_normalize_numerical_features()
        except Exception:
            # May fail due to single sample
            pass


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity throughout pipeline"""
    
    def setUp(self):
        # Create test data
        self.data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'cat_col': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        self.pipeline = DataPreprocessingPipeline()
        self.pipeline.raw_data = self.data.copy()
    
    def test_no_rows_lost(self):
        """Test that no rows are lost during preprocessing"""
        original_rows = len(self.pipeline.raw_data)
        
        self.pipeline.step1_handle_missing_values()
        self.assertEqual(len(self.pipeline.raw_data), original_rows)
        
        self.pipeline.step2_encode_categorical_variables()
        self.assertEqual(len(self.pipeline.raw_data), original_rows)
        
        self.pipeline.step3_engineer_features()
        self.assertEqual(len(self.pipeline.raw_data), original_rows)
    
    def test_column_names_preserved(self):
        """Test that important column names are preserved"""
        original_cols = set(self.pipeline.raw_data.columns)
        
        self.pipeline.step1_handle_missing_values()
        self.pipeline.step2_encode_categorical_variables()
        
        # Original columns should be present (or renamed)
        current_cols = set(self.pipeline.raw_data.columns)
        self.assertGreaterEqual(len(current_cols), len(original_cols))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return result
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run all tests
    success = run_tests()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
