"""
Test scenario analysis module.
Production-ready tests for scenario modeling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.scenarios import ScenarioAnalyzer


def test_scenario_analyzer_initialization():
    """Test scenario analyzer initialization."""
    analyzer = ScenarioAnalyzer()
    assert analyzer.config is not None
    assert analyzer.financial_planner is not None


def test_run_scenario_analysis(sample_forecast):
    """Test running scenario analysis."""
    analyzer = ScenarioAnalyzer()
    results = analyzer.run_scenario_analysis(sample_forecast, ['base', 'stress'])
    
    assert 'base' in results
    assert 'stress' in results
    
    # Check structure
    for scenario, plan in results.items():
        assert isinstance(plan, pd.DataFrame)
        assert len(plan) == len(sample_forecast)
        assert 'scenario' in plan.columns


def test_scenario_adjustments(sample_forecast):
    """Test scenario parameter adjustments."""
    analyzer = ScenarioAnalyzer()
    
    # Test stress scenario adjustments
    stress_config = {'sales_change': -0.2, 'fx_change': 0.3, 'opex_change': 0.1}
    
    adjusted_sales, adjusted_opex, adjusted_fx = analyzer._apply_scenario_adjustments(
        sample_forecast, stress_config
    )
    
    # Sales should be 20% lower
    expected_sales = sample_forecast * 0.8
    pd.testing.assert_series_equal(adjusted_sales, expected_sales)
    
    # FX should be 30% higher
    assert adjusted_fx.iloc[0] == 30.0 * 1.3


def test_create_scenario_comparison(sample_forecast):
    """Test scenario comparison creation."""
    analyzer = ScenarioAnalyzer()
    results = analyzer.run_scenario_analysis(sample_forecast)
    comparison = analyzer.create_scenario_comparison()
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) > 0
    assert 'scenario' in comparison.columns
    assert 'total_sales' in comparison.columns
    assert 'total_fcf' in comparison.columns


def test_scenario_insights(sample_forecast):
    """Test scenario insights generation."""
    analyzer = ScenarioAnalyzer()
    analyzer.run_scenario_analysis(sample_forecast)
    insights = analyzer.get_scenario_insights()
    
    assert 'best_case_scenario' in insights
    assert 'worst_case_scenario' in insights
    assert 'risk_level' in insights
    assert insights['risk_level'] in ['Low', 'Medium', 'High']


def test_get_scenario_summary(sample_forecast):
    """Test individual scenario summary."""
    analyzer = ScenarioAnalyzer()
    results = analyzer.run_scenario_analysis(sample_forecast, ['base'])
    summary = analyzer.get_scenario_summary('base')
    
    assert 'scenario_name' in summary
    assert 'total_sales' in summary
    assert 'total_free_cash_flow' in summary
    assert summary['scenario_name'] == 'base'


def test_save_scenario_results(sample_forecast, tmp_path):
    """Test saving scenario results."""
    analyzer = ScenarioAnalyzer()
    results = analyzer.run_scenario_analysis(sample_forecast, ['base'])
    
    # Create scenarios subdirectory in tmp_path
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)
    
    # Mock output directory
    analyzer.config.get_output_path = lambda x: str(tmp_path / x)
    
    saved_files = analyzer.save_scenario_results()
    
    assert len(saved_files) > 0
    for file_path in saved_files.values():
        assert Path(file_path).exists()


def test_invalid_scenario_name():
    """Test handling of invalid scenario names."""
    analyzer = ScenarioAnalyzer()
    
    with pytest.raises(Exception):  # Should raise ConfigurationError
        analyzer.config.get_scenario('nonexistent_scenario')