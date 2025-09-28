"""
Configuration management for Elips Financial Planner.

This module provides configuration loading, validation, and access
for all system parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class Config:
    """Configuration manager for the financial planning system."""
    
    def __init__(self, config_path: Optional[str] = None, validate_on_load: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            validate_on_load: Whether to validate configuration on load.
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path or self._get_default_config_path()
        self.load()
        if validate_on_load:
            self.validate()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "config" / "settings.yaml")
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {self._config_path}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self._config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['forecast', 'finance']
        optional_sections = ['scenarios', 'risk', 'data', 'logging', 'output']
        
        # Check required sections
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate specific parameters if sections exist
        self._validate_forecast_config()
        self._validate_finance_config()
        
        if 'risk' in self._config:
            self._validate_risk_config()
        
        logging.info("Configuration validation passed")
    
    def _validate_forecast_config(self) -> None:
        """Validate forecasting configuration."""
        forecast = self._config['forecast']
        
        if 'model' not in forecast:
            raise ConfigurationError("Missing required forecast parameter: model")
        if forecast['model'] not in ['sarima', 'ets', 'prophet', 'auto']:
            raise ConfigurationError(f"Invalid forecast model: {forecast['model']}")
        
        if 'horizon' not in forecast:
            raise ConfigurationError("Missing required forecast parameter: horizon")
        if not 1 <= forecast['horizon'] <= 60:
            raise ConfigurationError(f"Forecast horizon must be 1-60 months: {forecast['horizon']}")
        
        if 'confidence_level' in forecast:
            if not 0.8 <= forecast['confidence_level'] <= 0.99:
                raise ConfigurationError(f"Confidence level must be 0.8-0.99: {forecast['confidence_level']}")
    
    def _validate_finance_config(self) -> None:
        """Validate financial configuration."""
        finance = self._config['finance']
        
        if 'tax_rate' in finance:
            if not 0.0 <= finance['tax_rate'] <= 1.0:
                raise ConfigurationError(f"Tax rate must be 0.0-1.0: {finance['tax_rate']}")
        
        if 'cogs_share' in finance:
            if not 0.0 <= finance['cogs_share'] <= 1.0:
                raise ConfigurationError(f"COGS share must be 0.0-1.0: {finance['cogs_share']}")
        
        if all(key in finance for key in ['dso', 'dpo', 'dio']):
            if finance['dso'] <= 0 or finance['dpo'] <= 0 or finance['dio'] <= 0:
                raise ConfigurationError("Working capital days must be positive")
    
    def _validate_risk_config(self) -> None:
        """Validate risk analysis configuration."""
        risk = self._config['risk']
        
        if not 100 <= risk['mc_runs'] <= 100000:
            raise ConfigurationError(f"MC runs must be 100-100000: {risk['mc_runs']}")
        
        if risk['sales_vol'] <= 0 or risk['fx_vol'] <= 0:
            raise ConfigurationError("Volatility parameters must be positive")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'finance.tax_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigurationError(f"Configuration key not found: {key}")
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'finance.tax_rate')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get scenario configuration.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Scenario configuration dictionary
        """
        scenarios = self.get('scenarios')
        if scenario_name not in scenarios:
            available = list(scenarios.keys())
            raise ConfigurationError(f"Unknown scenario '{scenario_name}'. Available: {available}")
        
        return scenarios[scenario_name]
    
    def list_scenarios(self) -> list:
        """Get list of available scenario names."""
        return list(self.get('scenarios').keys())
    
    def get_data_path(self, filename: str) -> str:
        """
        Get full path to data file.
        
        Args:
            filename: Data filename
            
        Returns:
            Full path to data file
        """
        project_root = Path(self._config_path).parent.parent
        return str(project_root / filename)
    
    def get_output_path(self, filename: str) -> str:
        """
        Get full path to output file.
        
        Args:
            filename: Output filename
            
        Returns:
            Full path to output file
        """
        project_root = Path(self._config_path).parent.parent
        output_dir = self.get('data.output_dir', 'outputs')
        output_path = project_root / output_dir
        output_path.mkdir(exist_ok=True)
        return str(output_path / filename)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(file={self._config_path}, sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload configuration (useful for testing or config changes).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        New configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config