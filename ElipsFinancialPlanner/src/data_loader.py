"""
Data loading and validation module for Elips Financial Planner.

This module handles loading historical financial data, validation,
and preprocessing for the financial planning system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, date
import logging

from .config import get_config, ConfigurationError
from .logger import get_logger, ErrorHandler


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataLoader:
    """
    Data loader and validator for financial planning system.
    
    Handles loading, validation, and preprocessing of historical
    financial data for forecasting and analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self._data: Optional[pd.DataFrame] = None
        
    def load_sales_history(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical sales data from CSV file.
        
        Args:
            file_path: Path to CSV file (if None, uses config default)
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            DataValidationError: If data validation fails
        """
        if file_path is None:
            file_path = self.config.get_data_path(
                self.config.get('data.input_file', 'data/sales_history.csv')
            )
        
        try:
            self.logger.info(f"Loading sales history from: {file_path}")
            
            # Load CSV with proper date parsing
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime with flexible parsing
            if 'date' in df.columns:
                try:
                    # Handle mixed date formats by trying different approaches
                    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')
                    
                    # If some dates still failed, try with mixed format
                    if df['date'].isna().any():
                        self.logger.info("Some dates failed initial parsing, trying mixed format...")
                        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                    
                    # Final check - if any dates still failed, log but continue
                    failed_dates = df['date'].isna().sum()
                    if failed_dates > 0:
                        self.logger.warning(f"{failed_dates} dates could not be parsed and will be dropped")
                        # Drop rows with unparseable dates
                        df = df.dropna(subset=['date'])
                        
                except Exception as e:
                    self.logger.warning(f"Date parsing failed: {e}")
                    raise DataValidationError(f"Could not parse date column: {e}")
            else:
                raise DataValidationError("Date column not found in data")
            
            self.logger.info(f"Loaded {len(df)} records from {Path(file_path).name}")
            
            # Validate data
            self._validate_data(df)
            
            # Store cleaned data
            self._data = self._preprocess_data(df)
            
            self.logger.info("Data loading and validation completed successfully")
            return self._data.copy()
            
        except FileNotFoundError:
            raise DataValidationError(f"Data file not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"Data file is empty: {file_path}")
        except Exception as e:
            ErrorHandler.handle_error(e, "data loading", self.logger)
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate loaded data against business rules.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        self.logger.info("Validating data structure and content...")
        
        # Check required columns
        required_columns = ['date', 'sales_total_try', 'opex_try', 'usdtry']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise DataValidationError("Date column must be datetime type")
        
        # Check for negative values where not allowed
        numeric_columns = ['sales_total_try', 'opex_try', 'usdtry']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(f"Column {col} must be numeric")
            
            min_value = self.config.get('data.min_sales', 0) if col == 'sales_total_try' else 0
            if (df[col] < min_value).any():
                raise DataValidationError(f"Column {col} contains values below {min_value}")
        
        # Check date range
        min_date = pd.to_datetime(self.config.get('data.min_date', '2018-01-01'))
        if (df['date'] < min_date).any():
            raise DataValidationError(f"Dates before {min_date} not allowed")
        
        # Check FX rate reasonableness
        max_fx = self.config.get('data.max_fx_rate', 50.0)
        if (df['usdtry'] > max_fx).any():
            raise DataValidationError(f"USD/TRY rate above {max_fx} seems unrealistic")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            self.logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check data continuity (monthly data should be continuous)
        df_sorted = df.sort_values('date')
        date_diff = df_sorted['date'].diff().dropna()
        expected_diff = pd.Timedelta(days=30)  # Approximate monthly
        
        large_gaps = date_diff > pd.Timedelta(days=45)  # More than 45 days
        if large_gaps.any():
            gap_count = large_gaps.sum()
            self.logger.warning(f"Found {gap_count} large gaps in date sequence")
        
        self.logger.info("Data validation completed successfully")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and clean the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # Handle missing values with intelligent interpolation
        numeric_columns = ['sales_total_try', 'opex_try', 'usdtry']
        
        # Enhanced sales data cleaning with multiple fallbacks
        if 'sales_total_try' in df_clean.columns:
            sales_col = df_clean['sales_total_try']
            initial_missing = sales_col.isna().sum()
            
            if initial_missing > 0:
                self.logger.info(f"Cleaning {initial_missing} missing sales values")
                
                # Strategy 1: Time-based interpolation
                try:
                    df_clean['sales_total_try'] = sales_col.interpolate(method='time', limit_direction='both')
                    remaining_missing = df_clean['sales_total_try'].isna().sum()
                    if remaining_missing > 0:
                        raise ValueError("Time interpolation incomplete")
                except:
                    # Strategy 2: Linear interpolation
                    try:
                        df_clean['sales_total_try'] = sales_col.interpolate(method='linear', limit_direction='both')
                        remaining_missing = df_clean['sales_total_try'].isna().sum()
                        if remaining_missing > 0:
                            raise ValueError("Linear interpolation incomplete")
                    except:
                        # Strategy 3: Forward/backward fill
                        df_clean['sales_total_try'] = sales_col.ffill().bfill()
                        remaining_missing = df_clean['sales_total_try'].isna().sum()
                        if remaining_missing > 0:
                            # Strategy 4: Use median of available data
                            median_val = sales_col.median()
                            if pd.isna(median_val) or median_val <= 0:
                                median_val = 2500000  # Default 2.5M TRY
                            df_clean['sales_total_try'] = df_clean['sales_total_try'].fillna(median_val)
                
                final_missing = df_clean['sales_total_try'].isna().sum()
                self.logger.info(f"Sales cleaning completed: {initial_missing} -> {final_missing} missing values")
        
        # For other columns: forward fill then backward fill
        other_cols = [col for col in numeric_columns if col != 'sales_total_try']
        for col in other_cols:
            df_clean[col] = df_clean[col].ffill().bfill()
        
        # Add derived columns
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['quarter'] = df_clean['date'].dt.quarter
        
        # Calculate month-over-month growth rates
        df_clean['sales_growth_mom'] = df_clean['sales_total_try'].pct_change()
        df_clean['fx_change_mom'] = df_clean['usdtry'].pct_change()
        
        # Calculate year-over-year growth rates (12-month lag)
        df_clean['sales_growth_yoy'] = df_clean['sales_total_try'].pct_change(periods=12)
        
        # Add seasonal indicators
        df_clean['is_q4'] = (df_clean['quarter'] == 4).astype(int)
        df_clean['is_year_end'] = (df_clean['month'] == 12).astype(int)
        
        self.logger.info(f"Preprocessing completed. Final dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        return df_clean
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_sales_history() first.")
        
        summary = {
            'total_records': len(self._data),
            'date_range': {
                'start': self._data['date'].min().strftime('%Y-%m-%d'),
                'end': self._data['date'].max().strftime('%Y-%m-%d')
            },
            'sales_statistics': {
                'mean': float(self._data['sales_total_try'].mean()),
                'median': float(self._data['sales_total_try'].median()),
                'std': float(self._data['sales_total_try'].std()),
                'min': float(self._data['sales_total_try'].min()),
                'max': float(self._data['sales_total_try'].max())
            },
            'fx_statistics': {
                'mean': float(self._data['usdtry'].mean()),
                'median': float(self._data['usdtry'].median()),
                'min': float(self._data['usdtry'].min()),
                'max': float(self._data['usdtry'].max())
            },
            'missing_values': self._data.isnull().sum().to_dict(),
            'data_quality': self._assess_data_quality()
        }
        
        return summary
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """
        Assess data quality metrics.
        
        Returns:
            Dictionary with data quality assessment
        """
        if self._data is None:
            return {}
        
        # Calculate completeness
        completeness = (1 - self._data.isnull().sum() / len(self._data)) * 100
        
        # Check for outliers (using IQR method)
        def detect_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        
        outliers = {
            'sales_outliers': detect_outliers(self._data['sales_total_try']),
            'fx_outliers': detect_outliers(self._data['usdtry'])
        }
        
        # Data consistency checks
        consistency = {
            'monotonic_dates': self._data['date'].is_monotonic_increasing,
            'positive_sales': (self._data['sales_total_try'] >= 0).all(),
            'reasonable_fx': (self._data['usdtry'] > 0).all() and (self._data['usdtry'] < 100).all()
        }
        
        return {
            'completeness_percent': completeness.to_dict(),
            'outliers': outliers,
            'consistency_checks': consistency,
            'overall_quality': 'Good' if all(consistency.values()) and completeness.min() > 95 else 'Needs Review'
        }
    
    def save_processed_data(self, file_path: Optional[str] = None) -> str:
        """
        Save processed data to CSV file.
        
        Args:
            file_path: Output file path (if None, uses default)
            
        Returns:
            Path to saved file
        """
        if self._data is None:
            raise ValueError("No data to save. Call load_sales_history() first.")
        
        if file_path is None:
            file_path = self.config.get_output_path("processed_sales_data.csv")
        
        self._data.to_csv(file_path, index=False)
        self.logger.info(f"Processed data saved to: {file_path}")
        
        return file_path
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data (read-only copy)."""
        return self._data.copy() if self._data is not None else None
    
    def get_time_series(self, column: str = 'sales_total_try') -> pd.Series:
        """
        Get time series for specified column.
        
        Args:
            column: Column name to extract
            
        Returns:
            Time series with date index
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_sales_history() first.")
        
        if column not in self._data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        ts = self._data.set_index('date')[column]
        return ts
    
    def split_train_test(
        self, 
        test_periods: Optional[int] = None,
        test_start_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            test_periods: Number of periods for test set
            test_start_date: Start date for test set (YYYY-MM-DD)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_sales_history() first.")
        
        if test_periods is None:
            test_periods = self.config.get('forecast.validation_periods', 6)
        
        if test_start_date:
            split_date = pd.to_datetime(test_start_date)
            train_df = self._data[self._data['date'] < split_date].copy()
            test_df = self._data[self._data['date'] >= split_date].copy()
        else:
            # Use last N periods for testing
            split_idx = len(self._data) - test_periods
            train_df = self._data.iloc[:split_idx].copy()
            test_df = self._data.iloc[split_idx:].copy()
        
        self.logger.info(f"Data split: {len(train_df)} training, {len(test_df)} testing records")
        return train_df, test_df