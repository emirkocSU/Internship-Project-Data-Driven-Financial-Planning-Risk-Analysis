"""
Test data loading and validation.
Production-ready tests for data operations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import DataLoader, DataValidationError


def test_data_loader_initialization(data_loader):
    """Test data loader initialization."""
    assert data_loader.config is not None
    assert data_loader._data is None


def test_load_sales_history(data_loader, sample_csv_file):
    """Test loading sales history data."""
    data = data_loader.load_sales_history(sample_csv_file)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'date' in data.columns
    assert 'sales_total_try' in data.columns
    assert pd.api.types.is_datetime64_any_dtype(data['date'])


def test_data_validation_success(data_loader, sample_csv_file):
    """Test successful data validation."""
    # Should not raise any exceptions
    data_loader.load_sales_history(sample_csv_file)


def test_data_validation_missing_columns(data_loader, tmp_path):
    """Test validation with missing columns."""
    # Create invalid data
    invalid_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
        'sales_total_try': np.random.randn(10) * 1000000
        # Missing opex_try and usdtry
    })
    
    csv_file = tmp_path / "invalid.csv"
    invalid_data.to_csv(csv_file, index=False)
    
    with pytest.raises(DataValidationError):
        data_loader.load_sales_history(str(csv_file))


def test_data_validation_negative_values(data_loader, tmp_path):
    """Test validation with negative values."""
    invalid_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='MS'),
        'sales_total_try': [-1000000, 2000000, 1500000, 1800000, 2200000],
        'opex_try': [500000, 600000, 550000, 580000, 620000],
        'usdtry': [25, 26, 27, 28, 29]
    })
    
    csv_file = tmp_path / "negative.csv"
    invalid_data.to_csv(csv_file, index=False)
    
    with pytest.raises(DataValidationError):
        data_loader.load_sales_history(str(csv_file))


def test_data_preprocessing(data_loader, sample_csv_file):
    """Test data preprocessing features."""
    data = data_loader.load_sales_history(sample_csv_file)
    
    # Check derived columns
    assert 'year' in data.columns
    assert 'month' in data.columns
    assert 'quarter' in data.columns
    assert 'sales_growth_mom' in data.columns


def test_get_data_summary(data_loader, sample_csv_file):
    """Test data summary generation."""
    data_loader.load_sales_history(sample_csv_file)
    summary = data_loader.get_data_summary()
    
    assert 'total_records' in summary
    assert 'date_range' in summary
    assert 'sales_statistics' in summary
    assert 'data_quality' in summary
    
    assert summary['total_records'] > 0


def test_get_time_series(data_loader, sample_csv_file):
    """Test time series extraction."""
    data_loader.load_sales_history(sample_csv_file)
    ts = data_loader.get_time_series('sales_total_try')
    
    assert isinstance(ts, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(ts.index)
    assert len(ts) > 0


def test_split_train_test(data_loader, sample_csv_file):
    """Test train/test split functionality."""
    data_loader.load_sales_history(sample_csv_file)
    train_df, test_df = data_loader.split_train_test(test_periods=6)
    
    assert len(test_df) == 6
    assert len(train_df) > 0
    assert len(train_df) + len(test_df) == len(data_loader._data)


def test_save_processed_data(data_loader, sample_csv_file, tmp_path):
    """Test saving processed data."""
    data_loader.load_sales_history(sample_csv_file)
    output_file = tmp_path / "processed.csv"
    
    saved_path = data_loader.save_processed_data(str(output_file))
    
    assert Path(saved_path).exists()
    
    # Verify saved data
    saved_data = pd.read_csv(saved_path)
    assert len(saved_data) == len(data_loader._data)


def test_missing_file_error(data_loader):
    """Test error handling for missing files."""
    with pytest.raises(DataValidationError):
        data_loader.load_sales_history("nonexistent_file.csv")


def test_data_quality_assessment(data_loader, sample_csv_file):
    """Test data quality assessment."""
    data_loader.load_sales_history(sample_csv_file)
    summary = data_loader.get_data_summary()
    
    quality = summary['data_quality']
    assert 'completeness_percent' in quality
    assert 'consistency_checks' in quality
    assert 'overall_quality' in quality