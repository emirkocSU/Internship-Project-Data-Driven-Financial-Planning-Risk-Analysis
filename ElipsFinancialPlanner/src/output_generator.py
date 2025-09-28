"""
Output generation module for Elips Financial Planner.
Production-ready CSV and PNG chart generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from .config import get_config
from .logger import get_logger


class OutputGenerator:
    """
    Standardized output generation for reports and charts.
    Handles CSV exports and visualization creation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        
        # Set up plotting
        chart_style = self.config.get('output.chart_style', 'seaborn-v0_8')
        try:
            plt.style.use(chart_style)
        except:
            plt.style.use('seaborn-v0_8')  # fallback
        self.dpi = self.config.get('output.chart_dpi', 300)
        self.save_charts = self.config.get('output.save_charts', True)
    
    def generate_financial_report(
        self, 
        financial_plan: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> str:
        """Generate standardized financial report CSV."""
        
        if output_file is None:
            output_file = self.config.get_output_path('reports/financial_report.csv')
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Round numerical columns
        precision = self.config.get('output.csv_precision', 2)
        report_df = financial_plan.copy()
        
        numeric_cols = report_df.select_dtypes(include=[np.number]).columns
        report_df[numeric_cols] = report_df[numeric_cols].round(precision)
        
        # Save to CSV
        report_df.to_csv(output_file, index=False)
        self.logger.info(f"Financial report saved to: {output_file}")
        
        return output_file
    
    def create_forecast_chart(
        self, 
        historical_data: pd.Series,
        forecast_data: pd.Series,
        confidence_intervals: Optional[Dict[str, pd.Series]] = None,
        title: str = "Sales Forecast",
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Create forecast visualization chart."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Clean data from NaN values
        historical_clean = historical_data.dropna()
        forecast_clean = forecast_data.dropna()
        
        # Plot historical data
        if not historical_clean.empty:
            ax.plot(historical_clean.index, historical_clean.values, 
                   label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        if not forecast_clean.empty:
            ax.plot(forecast_clean.index, forecast_clean.values,
                   label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Add confidence intervals if provided
        if confidence_intervals:
            ax.fill_between(
                forecast_data.index,
                confidence_intervals['lower'].values,
                confidence_intervals['upper'].values,
                alpha=0.3, color='red', label='95% Confidence Interval'
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (TRY)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_charts and output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Forecast chart saved to: {output_file}")
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close()
            return None
    
    def create_scenario_comparison_chart(
        self, 
        scenario_results: Dict[str, pd.DataFrame],
        metric: str = 'free_cash_flow',
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Create scenario comparison chart."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (scenario_name, df) in enumerate(scenario_results.items()):
            color = colors[i % len(colors)]
            
            # Clean data and check if metric exists
            if metric in df.columns:
                data_clean = df[metric].fillna(0)
                date_clean = df['date']
                
                # Only plot if we have valid data
                if not data_clean.empty and not date_clean.empty:
                    ax.plot(date_clean, data_clean, 
                           label=scenario_name.replace('_', ' ').title(),
                           color=color, linewidth=2)
        
        ax.set_title(f'Scenario Comparison - {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (TRY)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if self.save_charts and output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Scenario comparison chart saved to: {output_file}")
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close()
            return None
    
    def create_risk_distribution_chart(
        self, 
        simulation_data: np.ndarray,
        var_95: float,
        cvar_95: float,
        title: str = "Risk Distribution",
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Create risk distribution visualization."""
        
        # Clean simulation data
        sim_data_clean = simulation_data[~np.isnan(simulation_data)]
        if len(sim_data_clean) == 0:
            self.logger.warning("No valid simulation data for risk chart")
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        ax1.hist(sim_data_clean, bins=50, alpha=0.7, color='skyblue', density=True)
        
        # Only add lines if values are finite
        if np.isfinite(var_95):
            ax1.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:,.0f}')
        if np.isfinite(cvar_95):
            ax1.axvline(cvar_95, color='darkred', linestyle='--', linewidth=2, label=f'CVaR 95%: {cvar_95:,.0f}')
        if np.isfinite(sim_data_clean.mean()):
            ax1.axvline(sim_data_clean.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {sim_data_clean.mean():,.0f}')
        
        ax1.set_title(f'{title} - Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Value (TRY)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(sim_data_clean, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel('Value (TRY)')
        ax2.set_title(f'{title} - Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_charts and output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Risk distribution chart saved to: {output_file}")
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close()
            return None
    
    def create_financial_dashboard(
        self, 
        financial_plan: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Create comprehensive financial dashboard."""
        
        # Clean financial plan from NaN values
        financial_plan_clean = financial_plan.fillna(0)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Elips Medikal - Financial Dashboard', fontsize=16, fontweight='bold')
        
        # Sales and gross profit
        axes[0, 0].plot(financial_plan_clean['date'], financial_plan_clean['sales'], 
                       label='Sales', color='blue', linewidth=2)
        axes[0, 0].plot(financial_plan_clean['date'], financial_plan_clean['gross_profit'], 
                       label='Gross Profit', color='green', linewidth=2)
        axes[0, 0].set_title('Revenue & Gross Profit')
        axes[0, 0].set_ylabel('TRY')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # EBIT and margins
        ax_margin = axes[0, 1].twinx()
        axes[0, 1].plot(financial_plan_clean['date'], financial_plan_clean['ebit'], 
                       color='purple', linewidth=2)
        axes[0, 1].set_ylabel('EBIT (TRY)', color='purple')
        
        # Avoid division by zero
        safe_sales = financial_plan_clean['sales'].replace(0, 1)
        margin_pct = (financial_plan_clean['ebit'] / safe_sales * 100)
        ax_margin.plot(financial_plan_clean['date'], margin_pct, 
                      color='orange', linewidth=2)
        ax_margin.set_ylabel('EBIT Margin (%)', color='orange')
        axes[0, 1].set_title('EBIT & Margin')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Free cash flow
        axes[1, 0].bar(financial_plan_clean['date'], financial_plan_clean['free_cash_flow'], 
                      color=['green' if x >= 0 else 'red' for x in financial_plan_clean['free_cash_flow']],
                      alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Free Cash Flow')
        axes[1, 0].set_ylabel('TRY')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Working capital
        axes[1, 1].plot(financial_plan_clean['date'], financial_plan_clean['working_capital'], 
                       color='brown', linewidth=2)
        axes[1, 1].set_title('Working Capital')
        axes[1, 1].set_ylabel('TRY')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_charts and output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Financial dashboard saved to: {output_file}")
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close()
            return None
    
    def generate_executive_summary(
        self, 
        financial_metrics: Dict[str, float],
        risk_metrics: Dict[str, float],
        scenario_insights: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """Generate executive summary report."""
        
        if output_file is None:
            output_file = self.config.get_output_path('reports/executive_summary.txt')
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        summary_text = f"""
ELIPS MEDIKAL FINANCIAL PLANNING - EXECUTIVE SUMMARY
====================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

FINANCIAL PROJECTIONS
---------------------
Total Sales (12M):        {financial_metrics.get('total_sales', 0):,.0f} TRY
Total Free Cash Flow:     {financial_metrics.get('total_free_cash_flow', 0):,.0f} TRY
Gross Margin:             {financial_metrics.get('gross_margin_percent', 0):.1f}%
EBIT Margin:              {financial_metrics.get('ebit_margin_percent', 0):.1f}%
Minimum Monthly FCF:      {financial_metrics.get('minimum_fcf', 0):,.0f} TRY

RISK ASSESSMENT
---------------
Overall Risk Level:       {risk_metrics.get('overall_risk_level', 'Unknown')}
VaR (95%):               {risk_metrics.get('worst_case_5pct', 0):,.0f} TRY
Expected Shortfall:       {risk_metrics.get('expected_shortfall', 0):,.0f} TRY
Probability of Loss:      {risk_metrics.get('probability_of_loss', 0):.1%}
Volatility:              {risk_metrics.get('volatility', 0):,.0f} TRY

SCENARIO ANALYSIS
-----------------
Best Case Scenario:       {scenario_insights.get('best_case_scenario', 'N/A')}
Worst Case Scenario:      {scenario_insights.get('worst_case_scenario', 'N/A')}
FCF Range:               {scenario_insights.get('fcf_range', 0):,.0f} TRY
Risk Level:              {scenario_insights.get('risk_level', 'Unknown')}

KEY RECOMMENDATIONS
-------------------
1. Monitor cash flow closely during periods of FX volatility
2. Maintain adequate liquidity buffers for working capital needs
3. Consider hedging strategies for USD exposure
4. Review pricing strategies to protect margins

====================================================
Report generated by Elips Financial Planner v{self.config.get('version', '0.1.0')}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        self.logger.info(f"Executive summary saved to: {output_file}")
        return output_file