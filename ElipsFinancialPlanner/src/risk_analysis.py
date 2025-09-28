"""
Monte Carlo risk analysis module for Elips Financial Planner.
Production-ready VaR/CVaR calculations and risk metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats

from .config import get_config
from .logger import get_logger
from .financial_planner import FinancialPlanner


class RiskAnalyzer:
    """
    Monte Carlo risk analysis engine.
    Calculates VaR, CVaR, and risk distributions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.financial_planner = FinancialPlanner(config_path)
        
        # Risk parameters
        self.mc_runs = self.config.get('risk.mc_runs', 1000)
        self.sales_vol = self.config.get('risk.sales_vol', 0.15)
        self.fx_vol = self.config.get('risk.fx_vol', 0.20)
        self.var_confidence = self.config.get('risk.var_confidence', 0.95)
        
        # Set random seed for reproducibility
        np.random.seed(self.config.get('random_seed', 42))
    
    def run_monte_carlo_analysis(
        self, 
        base_sales_forecast: pd.Series,
        base_opex: Optional[pd.Series] = None,
        base_fx: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for risk analysis.
        
        Args:
            base_sales_forecast: Base sales forecast
            base_opex: Base operating expenses
            base_fx: Base FX rates
            
        Returns:
            Monte Carlo simulation results
        """
        self.logger.info(f"Running Monte Carlo analysis with {self.mc_runs} simulations")
        
        # Initialize simulation results storage
        fcf_simulations = np.zeros((self.mc_runs, len(base_sales_forecast)))
        ebit_simulations = np.zeros((self.mc_runs, len(base_sales_forecast)))
        sales_simulations = np.zeros((self.mc_runs, len(base_sales_forecast)))
        
        # Run simulations
        for i in range(self.mc_runs):
            # Generate random shocks
            sales_shocks = self._generate_sales_shocks(len(base_sales_forecast))
            fx_shocks = self._generate_fx_shocks(len(base_sales_forecast))
            
            # Apply shocks to base forecasts
            shocked_sales = base_sales_forecast * (1 + sales_shocks)
            shocked_fx = self._apply_fx_shocks(base_fx, fx_shocks) if base_fx is not None else None
            
            # Generate financial plan for this simulation
            financial_plan = self.financial_planner.create_financial_plan(
                shocked_sales, base_opex, shocked_fx
            )
            
            # Store results
            fcf_simulations[i, :] = financial_plan['free_cash_flow'].values
            ebit_simulations[i, :] = financial_plan['ebit'].values
            sales_simulations[i, :] = financial_plan['sales'].values
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(fcf_simulations, ebit_simulations)
        
        # Compile results
        results = {
            'simulation_params': {
                'mc_runs': self.mc_runs,
                'sales_volatility': self.sales_vol,
                'fx_volatility': self.fx_vol,
                'confidence_level': self.var_confidence
            },
            'risk_metrics': risk_metrics,
            'fcf_distribution': self._analyze_distribution(fcf_simulations),
            'ebit_distribution': self._analyze_distribution(ebit_simulations),
            'simulation_data': {
                'fcf_simulations': fcf_simulations,
                'ebit_simulations': ebit_simulations,
                'sales_simulations': sales_simulations
            }
        }
        
        self.logger.info("Monte Carlo analysis completed successfully")
        return results
    
    def _generate_sales_shocks(self, periods: int) -> np.ndarray:
        """Generate correlated sales shocks."""
        
        # Generate AR(1) process for sales shocks
        ar_coeff = 0.3  # Autocorrelation coefficient
        shocks = np.zeros(periods)
        shocks[0] = np.random.normal(0, self.sales_vol)
        
        for t in range(1, periods):
            shocks[t] = ar_coeff * shocks[t-1] + np.random.normal(0, self.sales_vol * np.sqrt(1 - ar_coeff**2))
        
        return shocks
    
    def _generate_fx_shocks(self, periods: int) -> np.ndarray:
        """Generate FX rate shocks."""
        
        # FX shocks with higher persistence
        ar_coeff = 0.7  # Higher persistence for FX
        shocks = np.zeros(periods)
        shocks[0] = np.random.normal(0, self.fx_vol)
        
        for t in range(1, periods):
            shocks[t] = ar_coeff * shocks[t-1] + np.random.normal(0, self.fx_vol * np.sqrt(1 - ar_coeff**2))
        
        return shocks
    
    def _apply_fx_shocks(self, base_fx: pd.Series, fx_shocks: np.ndarray) -> pd.Series:
        """Apply FX shocks to base rates."""
        return base_fx * (1 + fx_shocks)
    
    def _calculate_risk_metrics(self, fcf_sims: np.ndarray, ebit_sims: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        
        # Total FCF over planning horizon
        total_fcf = fcf_sims.sum(axis=1)
        total_ebit = ebit_sims.sum(axis=1)
        
        # Minimum monthly FCF across simulations
        min_monthly_fcf = fcf_sims.min(axis=1)
        
        # VaR and CVaR calculations
        var_level = 1 - self.var_confidence
        fcf_var = np.percentile(total_fcf, var_level * 100)
        fcf_cvar = total_fcf[total_fcf <= fcf_var].mean()
        
        ebit_var = np.percentile(total_ebit, var_level * 100)
        ebit_cvar = total_ebit[total_ebit <= ebit_var].mean()
        
        # Risk of negative cash flows
        prob_negative_fcf = (min_monthly_fcf < 0).mean()
        prob_negative_total_fcf = (total_fcf < 0).mean()
        
        return {
            'fcf_var_95': float(fcf_var),
            'fcf_cvar_95': float(fcf_cvar),
            'ebit_var_95': float(ebit_var),
            'ebit_cvar_95': float(ebit_cvar),
            'fcf_mean': float(total_fcf.mean()),
            'fcf_std': float(total_fcf.std()),
            'fcf_min': float(total_fcf.min()),
            'fcf_max': float(total_fcf.max()),
            'prob_negative_monthly_fcf': float(prob_negative_fcf),
            'prob_negative_total_fcf': float(prob_negative_total_fcf),
            'fcf_coefficient_of_variation': float(total_fcf.std() / total_fcf.mean()) if total_fcf.mean() != 0 else float('inf')
        }
    
    def _analyze_distribution(self, simulations: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical distribution of simulation results."""
        
        total_values = simulations.sum(axis=1)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f'p{p}': float(np.percentile(total_values, p)) for p in percentiles}
        
        # Distribution tests
        try:
            # Normality test
            _, normality_p = stats.normaltest(total_values)
            is_normal = normality_p > 0.05
            
            # Skewness and kurtosis
            skewness = float(stats.skew(total_values))
            kurtosis = float(stats.kurtosis(total_values))
            
        except Exception:
            is_normal = False
            skewness = 0.0
            kurtosis = 0.0
            normality_p = 0.0
        
        return {
            'percentiles': percentile_values,
            'mean': float(total_values.mean()),
            'std': float(total_values.std()),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': is_normal,
            'normality_p_value': float(normality_p)
        }
    
    def calculate_stress_scenarios(self, base_forecast: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate deterministic stress scenarios."""
        
        stress_scenarios = {
            'severe_downturn': {'sales_shock': -0.30, 'fx_shock': 0.40},
            'moderate_stress': {'sales_shock': -0.15, 'fx_shock': 0.20},
            'fx_crisis': {'sales_shock': -0.10, 'fx_shock': 0.50},
            'demand_shock': {'sales_shock': -0.25, 'fx_shock': 0.10}
        }
        
        results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            # Apply uniform shocks
            stressed_sales = base_forecast * (1 + shocks['sales_shock'])
            stressed_fx = pd.Series(30.0 * (1 + shocks['fx_shock']), index=base_forecast.index)
            
            # Generate financial plan
            financial_plan = self.financial_planner.create_financial_plan(
                stressed_sales, None, stressed_fx
            )
            
            # Calculate metrics
            total_fcf = financial_plan['free_cash_flow'].sum()
            min_monthly_fcf = financial_plan['free_cash_flow'].min()
            
            results[scenario_name] = {
                'total_fcf': float(total_fcf),
                'min_monthly_fcf': float(min_monthly_fcf),
                'sales_impact': shocks['sales_shock'],
                'fx_impact': shocks['fx_shock']
            }
        
        return results
    
    def save_risk_analysis(self, mc_results: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """Save risk analysis results."""
        
        if file_path is None:
            file_path = self.config.get_output_path('risk_analysis_results.json')
        
        # Prepare serializable results (exclude simulation arrays)
        export_results = {
            'simulation_params': mc_results['simulation_params'],
            'risk_metrics': mc_results['risk_metrics'],
            'fcf_distribution': mc_results['fcf_distribution'],
            'ebit_distribution': mc_results['ebit_distribution'],
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(file_path, 'w') as f:
            json.dump(export_results, f, indent=2, default=str)
        
        self.logger.info(f"Risk analysis results saved to: {file_path}")
        return file_path
    
    def get_risk_summary(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive risk summary."""
        
        metrics = mc_results['risk_metrics']
        
        # Risk level assessment
        cv = metrics['fcf_coefficient_of_variation']
        prob_negative = metrics['prob_negative_total_fcf']
        
        if cv > 0.5 or prob_negative > 0.2:
            risk_level = 'High'
        elif cv > 0.3 or prob_negative > 0.1:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'overall_risk_level': risk_level,
            'expected_total_fcf': metrics['fcf_mean'],
            'worst_case_5pct': metrics['fcf_var_95'],
            'expected_shortfall': metrics['fcf_cvar_95'],
            'probability_of_loss': metrics['prob_negative_total_fcf'],
            'volatility': metrics['fcf_std'],
            'risk_adjusted_return': metrics['fcf_mean'] / metrics['fcf_std'] if metrics['fcf_std'] > 0 else 0
        }