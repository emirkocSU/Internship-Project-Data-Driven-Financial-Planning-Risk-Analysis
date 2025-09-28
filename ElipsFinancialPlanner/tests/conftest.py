"""
Pytest configuration and shared fixtures.
Production-ready test setup for Elips Financial Planner.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.config import Config
from src.data_loader import DataLoader
from src.financial_planner import FinancialPlanner


@pytest.fixture
def test_config():
    """Create test configuration."""
    config_data = {
        'random_seed': 42,
        'forecast': {
            'horizon': 6, 
            'model': 'sarima', 
            'seasonal': True,
            'confidence_level': 0.95,  # Add missing confidence_level
            'validation_periods': 6
        },
        'finance': {
            'tax_rate': 0.22,
            'cogs_share': 0.65,
            'beta_fx': 0.15,
            'dso': 45,
            'dpo': 30,
            'dio': 60,
            'depreciation': 50000,
            'capex': 100000
        },
        'scenarios': {
            'base': {'sales_change': 0.0, 'fx_change': 0.0, 'opex_change': 0.0},
            'best': {'sales_change': 0.15, 'fx_change': 0.0, 'opex_change': 0.0},
            'fx_shock': {'sales_change': -0.05, 'fx_change': 0.25, 'opex_change': 0.0},
            'cost_cut': {'sales_change': 0.0, 'fx_change': 0.0, 'opex_change': -0.10},
            'stress': {'sales_change': -0.2, 'fx_change': 0.3, 'opex_change': 0.0}
        },
        'risk': {
            'mc_runs': 100, 
            'sales_vol': 0.15, 
            'fx_vol': 0.20,
            'var_confidence': 0.95,
            'time_horizon': 12
        },
        'data': {
            'input_file': 'data/sales_history.csv',
            'output_dir': 'outputs',
            'date_format': '%Y-%m-%d',
            'min_date': '2020-01-01', 
            'max_fx_rate': 50.0,
            'min_sales': 0
        },
        'logging': {
            'level': 'WARNING',
            'file_path': 'outputs/elips_planner.log',
            'max_file_size': 10485760,
            'backup_count': 5,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'output': {
            'csv_precision': 2,
            'chart_style': 'seaborn',
            'chart_dpi': 300,
            'save_charts': True
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = f.name
    
    config = Config(config_path)
    yield config
    
    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def sample_data():
    """Create sample financial data."""
    dates = pd.date_range('2020-01-01', periods=36, freq='MS')
    np.random.seed(42)
    
    # Generate realistic data
    base_sales = 2000000
    trend = np.linspace(0, 0.5, 36)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(36) / 12)
    noise = 0.05 * np.random.randn(36)
    
    sales = base_sales * (1 + trend + seasonal + noise)
    opex = sales * 0.25 * (1 + 0.02 * np.random.randn(36))
    usdtry = 25 + 5 * np.cumsum(0.02 * np.random.randn(36))
    
    return pd.DataFrame({
        'date': dates,
        'sales_total_try': sales,
        'opex_try': opex,
        'usdtry': usdtry
    })


@pytest.fixture
def sample_csv_file(sample_data, tmp_path):
    """Create temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def data_loader(test_config):
    """Create configured data loader."""
    return DataLoader()


@pytest.fixture
def financial_planner(test_config):
    """Create configured financial planner."""
    return FinancialPlanner()


@pytest.fixture
def sample_forecast():
    """Create sample forecast data."""
    dates = pd.date_range('2024-07-01', periods=12, freq='MS')
    np.random.seed(42)
    
    # Generate forecast with trend
    base_value = 3000000
    growth = np.linspace(0, 0.2, 12)
    forecast_values = base_value * (1 + growth + 0.05 * np.random.randn(12))
    
    return pd.Series(forecast_values, index=dates)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "charts").mkdir()
    (output_dir / "reports").mkdir()
    (output_dir / "scenarios").mkdir()
    return str(output_dir)