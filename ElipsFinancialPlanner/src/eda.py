"""
Exploratory Data Analysis module for Elips Financial Planner.

This module provides comprehensive EDA capabilities including
trend analysis, seasonality detection, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings
from datetime import datetime
import logging

from .config import get_config
from .logger import get_logger
from .data_loader import DataLoader

warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """
    Exploratory Data Analysis for financial time series data.
    
    Provides comprehensive analysis including trend detection,
    seasonality analysis, correlation analysis, and visualization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize EDA analyzer.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.data_loader = DataLoader(config_path)
        self._data: Optional[pd.DataFrame] = None
        
        # Set plotting style
        chart_style = self.config.get('output.chart_style', 'seaborn-v0_8')
        try:
            plt.style.use(chart_style)
        except:
            plt.style.use('seaborn-v0_8')  # fallback
        sns.set_palette("husl")
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for analysis.
        
        Args:
            file_path: Optional path to data file
            
        Returns:
            Loaded DataFrame
        """
        self._data = self.data_loader.load_sales_history(file_path)
        self.logger.info(f"Data loaded for EDA: {len(self._data)} records")
        return self._data
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Returns:
            Dictionary containing analysis results
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Generating comprehensive EDA summary report...")
        
        report = {
            'data_overview': self._get_data_overview(),
            'time_series_analysis': self._analyze_time_series(),
            'seasonality_analysis': self._analyze_seasonality(),
            'trend_analysis': self._analyze_trends(),
            'correlation_analysis': self._analyze_correlations(),
            'volatility_analysis': self._analyze_volatility(),
            'business_insights': self._extract_business_insights()
        }
        
        self.logger.info("EDA summary report generated successfully")
        return report
    
    def _get_data_overview(self) -> Dict[str, Any]:
        """Get basic data overview."""
        overview = {
            'shape': self._data.shape,
            'date_range': {
                'start': self._data['date'].min().strftime('%Y-%m-%d'),
                'end': self._data['date'].max().strftime('%Y-%m-%d'),
                'duration_months': len(self._data)
            },
            'descriptive_stats': {
                'sales': self._data['sales_total_try'].describe().to_dict(),
                'opex': self._data['opex_try'].describe().to_dict(),
                'usdtry': self._data['usdtry'].describe().to_dict()
            },
            'missing_values': self._data.isnull().sum().to_dict()
        }
        return overview
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """Analyze time series properties."""
        # Growth rates
        sales_growth = self._data['sales_growth_mom'].dropna()
        fx_growth = self._data['fx_change_mom'].dropna()
        
        # Volatility metrics
        sales_volatility = sales_growth.std()
        fx_volatility = fx_growth.std()
        
        # Trend detection using linear regression
        from scipy import stats
        x = np.arange(len(self._data))
        sales_slope, _, sales_r_value, sales_p_value, _ = stats.linregress(x, self._data['sales_total_try'])
        fx_slope, _, fx_r_value, fx_p_value, _ = stats.linregress(x, self._data['usdtry'])
        
        analysis = {
            'growth_rates': {
                'sales_monthly_mean': float(sales_growth.mean()),
                'sales_monthly_std': float(sales_growth.std()),
                'fx_monthly_mean': float(fx_growth.mean()),
                'fx_monthly_std': float(fx_growth.std())
            },
            'volatility': {
                'sales_volatility': float(sales_volatility),
                'fx_volatility': float(fx_volatility),
                'volatility_ratio': float(sales_volatility / fx_volatility) if fx_volatility != 0 else None
            },
            'trend_analysis': {
                'sales_trend_slope': float(sales_slope),
                'sales_trend_r_squared': float(sales_r_value ** 2),
                'sales_trend_p_value': float(sales_p_value),
                'fx_trend_slope': float(fx_slope),
                'fx_trend_r_squared': float(fx_r_value ** 2),
                'fx_trend_p_value': float(fx_p_value)
            }
        }
        
        return analysis
    
    def _analyze_seasonality(self) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        # Monthly seasonality
        monthly_avg = self._data.groupby('month')['sales_total_try'].mean()
        monthly_std = self._data.groupby('month')['sales_total_try'].std()
        
        # Quarterly seasonality
        quarterly_avg = self._data.groupby('quarter')['sales_total_try'].mean()
        
        # Year-end effect
        year_end_sales = self._data[self._data['month'] == 12]['sales_total_try'].mean()
        regular_sales = self._data[self._data['month'] != 12]['sales_total_try'].mean()
        year_end_effect = (year_end_sales - regular_sales) / regular_sales
        
        # Seasonal decomposition (if scipy is available)
        try:
            from scipy import signal
            
            # Detrend the data
            sales_detrended = signal.detrend(self._data['sales_total_try'])
            
            # Calculate seasonal strength
            seasonal_strength = np.std(sales_detrended) / np.std(self._data['sales_total_try'])
            
        except ImportError:
            seasonal_strength = None
        
        seasonality = {
            'monthly_patterns': {
                'averages': monthly_avg.to_dict(),
                'standard_deviations': monthly_std.to_dict(),
                'peak_month': int(monthly_avg.idxmax()),
                'trough_month': int(monthly_avg.idxmin())
            },
            'quarterly_patterns': {
                'averages': quarterly_avg.to_dict(),
                'q4_premium': float((quarterly_avg[4] - quarterly_avg.mean()) / quarterly_avg.mean()) if 4 in quarterly_avg else 0
            },
            'year_end_effect': {
                'effect_magnitude': float(year_end_effect),
                'december_vs_average': float(year_end_sales / regular_sales - 1)
            },
            'seasonal_strength': float(seasonal_strength) if seasonal_strength else None
        }
        
        return seasonality
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze long-term trends."""
        # Annual growth rates
        annual_data = self._data.groupby('year').agg({
            'sales_total_try': 'sum',
            'opex_try': 'sum',
            'usdtry': 'mean'
        }).reset_index()
        
        if len(annual_data) > 1:
            annual_sales_growth = annual_data['sales_total_try'].pct_change().dropna()
            avg_annual_growth = annual_sales_growth.mean()
            
            # Compound Annual Growth Rate (CAGR)
            years = len(annual_data) - 1
            if years > 0:
                cagr = (annual_data['sales_total_try'].iloc[-1] / annual_data['sales_total_try'].iloc[0]) ** (1/years) - 1
            else:
                cagr = 0
        else:
            avg_annual_growth = 0
            cagr = 0
        
        # Trend consistency
        positive_growth_months = (self._data['sales_growth_mom'] > 0).sum()
        total_growth_months = self._data['sales_growth_mom'].notna().sum()
        growth_consistency = positive_growth_months / total_growth_months if total_growth_months > 0 else 0
        
        trends = {
            'annual_growth': {
                'average_annual_growth': float(avg_annual_growth),
                'cagr': float(cagr),
                'years_analyzed': len(annual_data)
            },
            'growth_consistency': {
                'positive_growth_ratio': float(growth_consistency),
                'trend_direction': 'Upward' if avg_annual_growth > 0.05 else 'Downward' if avg_annual_growth < -0.05 else 'Stable'
            }
        }
        
        return trends
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        # Correlation matrix
        numeric_cols = ['sales_total_try', 'opex_try', 'usdtry']
        correlation_matrix = self._data[numeric_cols].corr()
        
        # Lagged correlations (sales vs FX)
        lagged_correlations = {}
        for lag in range(1, 7):  # 1-6 month lags
            if len(self._data) > lag:
                fx_lagged = self._data['usdtry'].shift(lag)
                corr = self._data['sales_total_try'].corr(fx_lagged)
                lagged_correlations[f'fx_lag_{lag}'] = float(corr) if not np.isnan(corr) else 0
        
        correlations = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'key_correlations': {
                'sales_fx': float(correlation_matrix.loc['sales_total_try', 'usdtry']),
                'sales_opex': float(correlation_matrix.loc['sales_total_try', 'opex_try']),
                'fx_opex': float(correlation_matrix.loc['usdtry', 'opex_try'])
            },
            'lagged_fx_correlations': lagged_correlations
        }
        
        return correlations
    
    def _analyze_volatility(self) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        # Rolling volatility (3-month window)
        window = 3
        sales_rolling_vol = self._data['sales_growth_mom'].rolling(window=window).std()
        fx_rolling_vol = self._data['fx_change_mom'].rolling(window=window).std()
        
        # Volatility clustering
        high_vol_threshold = sales_rolling_vol.quantile(0.75)
        high_vol_periods = (sales_rolling_vol > high_vol_threshold).sum()
        
        volatility = {
            'rolling_volatility': {
                'sales_mean': float(sales_rolling_vol.mean()),
                'sales_max': float(sales_rolling_vol.max()),
                'fx_mean': float(fx_rolling_vol.mean()),
                'fx_max': float(fx_rolling_vol.max())
            },
            'volatility_clustering': {
                'high_volatility_periods': int(high_vol_periods),
                'volatility_persistence': float(sales_rolling_vol.autocorr(lag=1)) if len(sales_rolling_vol.dropna()) > 1 else 0
            }
        }
        
        return volatility
    
    def _extract_business_insights(self) -> Dict[str, Any]:
        """Extract business-relevant insights."""
        # Peak performance periods
        top_sales_months = self._data.nlargest(3, 'sales_total_try')
        
        # Market conditions during high performance
        high_performance_fx = top_sales_months['usdtry'].mean()
        avg_fx = self._data['usdtry'].mean()
        
        # Growth acceleration periods
        growth_acceleration = self._data['sales_growth_mom'].diff()
        acceleration_periods = growth_acceleration.nlargest(3)
        
        insights = {
            'peak_performance': {
                'top_months': top_sales_months[['date', 'sales_total_try', 'usdtry']].to_dict('records'),
                'avg_fx_during_peaks': float(high_performance_fx),
                'fx_vs_average': float(high_performance_fx / avg_fx - 1)
            },
            'growth_patterns': {
                'strongest_acceleration_periods': acceleration_periods.index.tolist(),
                'growth_volatility_assessment': 'High' if self._data['sales_growth_mom'].std() > 0.15 else 'Moderate' if self._data['sales_growth_mom'].std() > 0.08 else 'Low'
            },
            'market_sensitivity': {
                'fx_sensitivity_level': 'High' if abs(self._analyze_correlations()['key_correlations']['sales_fx']) > 0.5 else 'Moderate' if abs(self._analyze_correlations()['key_correlations']['sales_fx']) > 0.3 else 'Low'
            }
        }
        
        return insights
    
    def create_visualizations(self, save_charts: bool = True) -> Dict[str, str]:
        """
        Create comprehensive visualization suite.
        
        Args:
            save_charts: Whether to save charts to files
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Creating EDA visualizations...")
        
        saved_files = {}
        
        # 1. Time series overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Elips Medikal - Financial Time Series Overview', fontsize=16, fontweight='bold')
        
        # Sales over time
        axes[0, 0].plot(self._data['date'], self._data['sales_total_try'], color='blue', linewidth=2)
        axes[0, 0].set_title('Sales Revenue Over Time')
        axes[0, 0].set_ylabel('Sales (TRY)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # USD/TRY rate
        axes[0, 1].plot(self._data['date'], self._data['usdtry'], color='red', linewidth=2)
        axes[0, 1].set_title('USD/TRY Exchange Rate')
        axes[0, 1].set_ylabel('USD/TRY')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Operating expenses
        axes[1, 0].plot(self._data['date'], self._data['opex_try'], color='green', linewidth=2)
        axes[1, 0].set_title('Operating Expenses Over Time')
        axes[1, 0].set_ylabel('OpEx (TRY)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Growth rates
        axes[1, 1].plot(self._data['date'], self._data['sales_growth_mom'] * 100, color='purple', linewidth=2)
        axes[1, 1].set_title('Month-over-Month Sales Growth')
        axes[1, 1].set_ylabel('Growth Rate (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            file_path = self.config.get_output_path('charts/time_series_overview.png')
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files['time_series_overview'] = file_path
        
        plt.show()
        plt.close()
        
        # 2. Seasonal analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Seasonal Analysis', fontsize=16, fontweight='bold')
        
        # Monthly averages
        monthly_avg = self._data.groupby('month')['sales_total_try'].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='skyblue')
        axes[0, 0].set_title('Average Sales by Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Sales (TRY)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Quarterly patterns
        quarterly_avg = self._data.groupby('quarter')['sales_total_try'].mean()
        axes[0, 1].bar(quarterly_avg.index, quarterly_avg.values, color='lightcoral')
        axes[0, 1].set_title('Average Sales by Quarter')
        axes[0, 1].set_xlabel('Quarter')
        axes[0, 1].set_ylabel('Average Sales (TRY)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Year-over-year growth
        if 'sales_growth_yoy' in self._data.columns:
            axes[1, 0].plot(self._data['date'], self._data['sales_growth_yoy'] * 100, color='orange', linewidth=2)
            axes[1, 0].set_title('Year-over-Year Growth Rate')
            axes[1, 0].set_ylabel('YoY Growth (%)')
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Boxplot by quarter
        quarter_data = [self._data[self._data['quarter'] == q]['sales_total_try'].values for q in [1, 2, 3, 4]]
        axes[1, 1].boxplot(quarter_data, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 1].set_title('Sales Distribution by Quarter')
        axes[1, 1].set_ylabel('Sales (TRY)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            file_path = self.config.get_output_path('charts/seasonal_analysis.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files['seasonal_analysis'] = file_path
        
        plt.show()
        plt.close()
        
        # 3. Correlation analysis
        plt.figure(figsize=(10, 8))
        
        # Create correlation matrix
        numeric_cols = ['sales_total_try', 'opex_try', 'usdtry']
        correlation_matrix = self._data[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix - Key Financial Metrics', fontsize=14, fontweight='bold')
        
        if save_charts:
            file_path = self.config.get_output_path('charts/correlation_analysis.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files['correlation_analysis'] = file_path
        
        plt.show()
        plt.close()
        
        self.logger.info(f"Created {len(saved_files)} visualization charts")
        return saved_files
    
    def save_eda_report(self, file_path: Optional[str] = None) -> str:
        """
        Save comprehensive EDA report to file.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to saved report
        """
        if file_path is None:
            file_path = self.config.get_output_path('reports/eda_report.json')
        
        report = self.generate_summary_report()
        
        # Convert numpy types to Python types for JSON serialization
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        report_serializable = convert_numpy_types(report)
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(report_serializable, f, indent=2, default=str)
        
        self.logger.info(f"EDA report saved to: {file_path}")
        return file_path