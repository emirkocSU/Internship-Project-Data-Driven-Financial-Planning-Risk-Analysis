"""
Scenario analysis module for Elips Financial Planner.
Production-ready scenario modeling and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from .config import get_config
from .logger import get_logger
from .financial_planner import FinancialPlanner


class ScenarioAnalyzer:
    """
    Scenario analysis engine for financial planning.
    Applies parameter changes and generates comparisons.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.financial_planner = FinancialPlanner(config_path)
        self._scenario_results: Dict[str, pd.DataFrame] = {}
    
    def run_scenario_analysis(
        self, 
        base_sales_forecast: pd.Series,
        scenarios: Optional[List[str]] = None,
        base_opex: Optional[pd.Series] = None,
        base_fx: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive scenario analysis.
        
        Args:
            base_sales_forecast: Base sales forecast
            scenarios: List of scenario names to run
            base_opex: Base operating expenses
            base_fx: Base FX rates
            
        Returns:
            Dictionary of scenario results
        """
        if scenarios is None:
            scenarios = self.config.list_scenarios()
        
        self.logger.info(f"Running scenario analysis for: {scenarios}")
        
        results = {}
        
        for scenario_name in scenarios:
            scenario_config = self.config.get_scenario(scenario_name)
            
            # Apply scenario adjustments
            adjusted_sales, adjusted_opex, adjusted_fx = self._apply_scenario_adjustments(
                base_sales_forecast, scenario_config, base_opex, base_fx
            )
            
            # Generate financial plan for scenario
            financial_plan = self.financial_planner.create_financial_plan(
                adjusted_sales, adjusted_opex, adjusted_fx
            )
            
            # Add scenario metadata
            financial_plan['scenario'] = scenario_name
            financial_plan['scenario_description'] = scenario_config['description']
            
            results[scenario_name] = financial_plan
            
        self._scenario_results = results
        self.logger.info(f"Scenario analysis completed for {len(results)} scenarios")
        return results
    
    def _apply_scenario_adjustments(
        self, 
        base_sales: pd.Series,
        scenario_config: Dict[str, Any],
        base_opex: Optional[pd.Series] = None,
        base_fx: Optional[pd.Series] = None
    ) -> tuple:
        """Apply scenario adjustments to base forecasts."""
        
        # Sales adjustments
        sales_change = scenario_config.get('sales_change', 0.0)
        adjusted_sales = base_sales * (1 + sales_change)
        
        # OpEx adjustments
        opex_change = scenario_config.get('opex_change', 0.0)
        if base_opex is not None:
            adjusted_opex = base_opex * (1 + opex_change)
        else:
            # Default OpEx calculation with adjustment
            base_opex_level = adjusted_sales.iloc[0] * 0.2
            adjusted_opex = pd.Series(
                base_opex_level * (1 + 0.05) ** np.arange(len(adjusted_sales)) * (1 + opex_change),
                index=adjusted_sales.index
            )
        
        # FX adjustments
        fx_change = scenario_config.get('fx_change', 0.0)
        if base_fx is not None:
            adjusted_fx = base_fx * (1 + fx_change)
        else:
            # Default FX with adjustment
            base_fx_rate = 30.0
            adjusted_fx = pd.Series(
                base_fx_rate * (1 + fx_change),
                index=adjusted_sales.index
            )
        
        return adjusted_sales, adjusted_opex, adjusted_fx
    
    def create_scenario_comparison(self) -> pd.DataFrame:
        """Create scenario comparison summary."""
        
        if not self._scenario_results:
            raise ValueError("No scenario results available. Run scenario analysis first.")
        
        comparison_data = []
        
        for scenario_name, financial_plan in self._scenario_results.items():
            metrics = self.financial_planner.calculate_key_metrics(financial_plan)
            
            comparison_data.append({
                'scenario': scenario_name,
                'total_sales': metrics['total_sales'],
                'total_fcf': metrics['total_free_cash_flow'],
                'gross_margin_pct': metrics['gross_margin_percent'],
                'ebit_margin_pct': metrics['ebit_margin_percent'],
                'min_fcf': metrics['minimum_fcf'],
                'fcf_volatility': metrics['fcf_volatility']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate differences from base case
        if 'base' in comparison_df['scenario'].values:
            base_idx = comparison_df[comparison_df['scenario'] == 'base'].index[0]
            base_sales = comparison_df.loc[base_idx, 'total_sales']
            base_fcf = comparison_df.loc[base_idx, 'total_fcf']
            
            comparison_df['sales_vs_base_pct'] = (comparison_df['total_sales'] / base_sales - 1) * 100
            comparison_df['fcf_vs_base_pct'] = (comparison_df['total_fcf'] / base_fcf - 1) * 100
        
        return comparison_df
    
    def get_scenario_insights(self) -> Dict[str, Any]:
        """Generate insights from scenario analysis."""
        
        if not self._scenario_results:
            raise ValueError("No scenario results available.")
        
        comparison = self.create_scenario_comparison()
        
        # Find best and worst scenarios
        best_fcf_scenario = comparison.loc[comparison['total_fcf'].idxmax(), 'scenario']
        worst_fcf_scenario = comparison.loc[comparison['total_fcf'].idxmin(), 'scenario']
        
        # Risk assessment
        fcf_range = comparison['total_fcf'].max() - comparison['total_fcf'].min()
        base_fcf = comparison[comparison['scenario'] == 'base']['total_fcf'].iloc[0] if 'base' in comparison['scenario'].values else comparison['total_fcf'].mean()
        risk_percentage = (fcf_range / base_fcf) * 100
        
        insights = {
            'best_case_scenario': best_fcf_scenario,
            'worst_case_scenario': worst_fcf_scenario,
            'fcf_range': float(fcf_range),
            'risk_percentage': float(risk_percentage),
            'scenarios_with_negative_fcf': list(comparison[comparison['min_fcf'] < 0]['scenario']),
            'risk_level': 'High' if risk_percentage > 50 else 'Medium' if risk_percentage > 25 else 'Low'
        }
        
        return insights
    
    def save_scenario_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save all scenario results to files."""
        
        if not self._scenario_results:
            raise ValueError("No scenario results to save.")
        
        saved_files = {}
        
        # Save individual scenario results
        for scenario_name, financial_plan in self._scenario_results.items():
            file_path = self.config.get_output_path(f'scenarios/{scenario_name}_financial_plan.csv')
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            self.financial_planner.save_financial_plan(financial_plan, file_path)
            saved_files[f'{scenario_name}_plan'] = file_path
        
        # Save comparison summary
        comparison = self.create_scenario_comparison()
        comparison_path = self.config.get_output_path('scenarios/scenario_comparison.csv')
        Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        saved_files['comparison'] = comparison_path
        
        # Save insights
        insights = self.get_scenario_insights()
        import json
        insights_path = self.config.get_output_path('scenarios/scenario_insights.json')
        Path(insights_path).parent.mkdir(parents=True, exist_ok=True)
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        saved_files['insights'] = insights_path
        
        self.logger.info(f"Scenario results saved: {len(saved_files)} files")
        return saved_files
    
    @property
    def scenario_results(self) -> Dict[str, pd.DataFrame]:
        """Get scenario results (read-only)."""
        return self._scenario_results.copy()
    
    def get_scenario_summary(self, scenario_name: str) -> Dict[str, Any]:
        """Get summary for specific scenario."""
        
        if scenario_name not in self._scenario_results:
            raise ValueError(f"Scenario '{scenario_name}' not found in results.")
        
        financial_plan = self._scenario_results[scenario_name]
        metrics = self.financial_planner.calculate_key_metrics(financial_plan)
        
        return {
            'scenario_name': scenario_name,
            'description': financial_plan['scenario_description'].iloc[0],
            'total_sales': metrics['total_sales'],
            'total_free_cash_flow': metrics['total_free_cash_flow'],
            'avg_monthly_fcf': metrics['total_free_cash_flow'] / len(financial_plan),
            'gross_margin_percent': metrics['gross_margin_percent'],
            'ebit_margin_percent': metrics['ebit_margin_percent'],
            'minimum_monthly_fcf': metrics['minimum_fcf'],
            'fcf_volatility': metrics['fcf_volatility']
        }