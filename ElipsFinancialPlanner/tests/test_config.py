"""
Test configuration module.
Production-ready tests for configuration loading and validation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.config import Config, ConfigurationError


def test_config_loading(test_config):
    """Test basic configuration loading."""
    assert test_config.get('random_seed') == 42
    assert test_config.get('forecast.horizon') == 6
    assert test_config.get('finance.tax_rate') == 0.22


def test_config_validation(test_config):
    """Test configuration validation."""
    # Should not raise any exceptions
    test_config.validate()


def test_invalid_config():
    """Test invalid configuration handling."""
    invalid_config = {
        'forecast': {'horizon': -5},  # Invalid
        'finance': {'tax_rate': 1.5}  # Invalid
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(invalid_config, f)
        config_path = f.name
    
    try:
        config = Config(config_path, validate_on_load=False)
        with pytest.raises(ConfigurationError):
            config.validate()
    finally:
        os.unlink(config_path)


def test_scenario_access(test_config):
    """Test scenario configuration access."""
    scenarios = test_config.list_scenarios()
    assert 'base' in scenarios
    assert 'stress' in scenarios
    
    base_scenario = test_config.get_scenario('base')
    assert base_scenario['sales_change'] == 0.0


def test_missing_config_file():
    """Test handling of missing configuration file."""
    with pytest.raises(ConfigurationError):
        Config('nonexistent_config.yaml')


def test_config_set_get():
    """Test configuration setting and getting."""
    config = Config()
    config.set('test.value', 123)
    assert config.get('test.value') == 123


def test_path_resolution(test_config, tmp_path):
    """Test file path resolution."""
    # Mock the config path for testing
    test_config._config_path = str(tmp_path / 'config.yaml')
    
    data_path = test_config.get_data_path('test.csv')
    output_path = test_config.get_output_path('test.txt')
    
    assert 'test.csv' in data_path
    assert 'test.txt' in output_path