"""
Test financial planning module.
Production-ready tests for financial calculations.
"""

import pytest
import pandas as pd
import numpy as np

from src.financial_planner import FinancialPlanner


def test_financial_planner_initialization(financial_planner):
    """Test financial planner initialization."""
    assert financial_planner.tax_rate == 0.22
    assert financial_planner.cogs_share == 0.65
    assert financial_planner.dso == 45


def test_create_financial_plan(financial_planner, sample_forecast):
    """Test financial plan creation."""
    plan = financial_planner.create_financial_plan(sample_forecast)
    
    # Check structure
    expected_columns = [
        'date', 'sales', 'cogs', 'gross_profit', 'opex',
        'ebit', 'tax', 'nopat', 'accounts_receivable',
        'accounts_payable', 'inventory', 'working_capital',
        'wc_change', 'depreciation', 'capex', 'free_cash_flow'
    ]
    
    for col in expected_columns:
        assert col in plan.columns
    
    assert len(plan) == len(sample_forecast)


def test_financial_calculations(financial_planner):
    """Test core financial calculations."""
    # Simple test case
    sales = np.array([1000000, 1100000])
    opex = np.array([200000, 220000])
    fx = np.array([30.0, 32.0])
    
    results = financial_planner._calculate_financials(sales, opex, fx)
    
    # Test COGS calculation
    expected_cogs_1 = 1000000 * 0.65 * 1.0  # No FX impact on first period
    assert np.isclose(results['cogs'][0], expected_cogs_1)
    
    # Test gross profit
    expected_gp_1 = sales[0] - results['cogs'][0]
    assert np.isclose(results['gross_profit'][0], expected_gp_1)
    
    # Test EBIT
    expected_ebit_1 = results['gross_profit'][0] - opex[0]
    assert np.isclose(results['ebit'][0], expected_ebit_1)


def test_key_metrics_calculation(financial_planner, sample_forecast):
    """Test key metrics calculation."""
    plan = financial_planner.create_financial_plan(sample_forecast)
    metrics = financial_planner.calculate_key_metrics(plan)
    
    # Check required metrics
    required_metrics = [
        'total_sales', 'total_free_cash_flow', 'gross_margin_percent',
        'ebit_margin_percent', 'fcf_volatility', 'minimum_fcf'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_working_capital_calculations(financial_planner):
    """Test working capital calculations."""
    sales = np.array([1000000])
    opex = np.array([200000])
    fx = np.array([30.0])
    
    results = financial_planner._calculate_financials(sales, opex, fx)
    
    # Check working capital components
    expected_ar = sales[0] * 45 / 30  # DSO = 45 days
    expected_ap = results['cogs'][0] * 30 / 30  # DPO = 30 days
    expected_inv = results['cogs'][0] * 60 / 30  # DIO = 60 days
    
    assert np.isclose(results['ar'][0], expected_ar)
    assert np.isclose(results['ap'][0], expected_ap)
    assert np.isclose(results['inventory'][0], expected_inv)


def test_fx_impact_on_cogs(financial_planner):
    """Test FX impact on cost of goods sold."""
    sales = np.array([1000000, 1000000])
    opex = np.array([200000, 200000])
    fx_stable = np.array([30.0, 30.0])
    fx_shock = np.array([30.0, 36.0])  # 20% FX increase
    
    results_stable = financial_planner._calculate_financials(sales, opex, fx_stable)
    results_shock = financial_planner._calculate_financials(sales, opex, fx_shock)
    
    # COGS should be higher in second period with FX shock
    assert results_shock['cogs'][1] > results_stable['cogs'][1]


def test_tax_calculation(financial_planner):
    """Test tax calculation logic."""
    # Positive EBIT case
    sales_pos = np.array([1000000])
    opex_pos = np.array([200000])
    fx = np.array([30.0])
    
    results_pos = financial_planner._calculate_financials(sales_pos, opex_pos, fx)
    expected_tax = results_pos['ebit'][0] * 0.22
    assert np.isclose(results_pos['tax'][0], expected_tax)
    
    # Negative EBIT case (no tax)
    sales_neg = np.array([100000])
    opex_neg = np.array([500000])
    
    results_neg = financial_planner._calculate_financials(sales_neg, opex_neg, fx)
    assert results_neg['tax'][0] == 0.0


def test_save_financial_plan(financial_planner, sample_forecast, tmp_path):
    """Test saving financial plan to file."""
    plan = financial_planner.create_financial_plan(sample_forecast)
    output_file = tmp_path / "test_plan.csv"
    
    saved_path = financial_planner.save_financial_plan(plan, str(output_file))
    
    assert output_file.exists()
    
    # Verify file content
    loaded_plan = pd.read_csv(output_file)
    assert len(loaded_plan) == len(plan)
    assert 'sales' in loaded_plan.columns