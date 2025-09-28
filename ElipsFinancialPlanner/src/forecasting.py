"""
Time series forecasting module for Elips Financial Planner.

This module provides multiple forecasting models including SARIMA,
Exponential Smoothing, and Prophet for sales prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import warnings
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path

# Core forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some forecasting features will be disabled.")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using basic error metrics.")

# Optional Prophet import
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from .config import get_config
from .logger import get_logger
from .data_loader import DataLoader

warnings.filterwarnings('ignore')


# Fallback functions if sklearn is not available
def _fallback_mean_absolute_error(y_true, y_pred):
    """Fallback MAE calculation."""
    return np.mean(np.abs(y_true - y_pred))

def _fallback_mean_squared_error(y_true, y_pred):
    """Fallback MSE calculation."""
    return np.mean((y_true - y_pred) ** 2)

def _fallback_mean_absolute_percentage_error(y_true, y_pred):
    """Fallback MAPE calculation."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class ForecastingError(Exception):
    """Raised when forecasting operations fail."""
    pass


class TimeSeriesForecaster:
    """
    Comprehensive time series forecasting system.
    
    Supports multiple models: SARIMA, ETS, and Prophet.
    Includes automatic model selection and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize forecaster.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = get_logger(__name__)
        self.data_loader = DataLoader(config_path)
        
        # Check library availability
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available. SARIMA and ETS models will be disabled.")
        
        self._models: Dict[str, Any] = {}
        self._fitted_models: Dict[str, Any] = {}
        self._model_performance: Dict[str, Dict[str, float]] = {}
        self._data: Optional[pd.DataFrame] = None
        self._time_series: Optional[pd.Series] = None
        
    def load_data(self, file_path: Optional[str] = None, target_column: str = 'sales_total_try') -> pd.Series:
        """
        Load and prepare data for forecasting.
        
        Args:
            file_path: Optional path to data file
            target_column: Target variable for forecasting
            
        Returns:
            Time series data
        """
        self._data = self.data_loader.load_sales_history(file_path)
        self._time_series = self.data_loader.get_time_series(target_column)
        
        self.logger.info(f"Loaded time series: {len(self._time_series)} observations")
        self.logger.info(f"Date range: {self._time_series.index.min()} to {self._time_series.index.max()}")
        
        return self._time_series
    
    def get_time_series(self, column: str = 'sales_total_try') -> pd.Series:
        """
        Get the loaded time series data.
        
        Args:
            column: Column name (for compatibility)
            
        Returns:
            Time series data
        """
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        return self._time_series
    
    def analyze_time_series(self) -> Dict[str, Any]:
        """
        Analyze time series properties for model selection.
        
        Returns:
            Dictionary with analysis results
        """
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        
        self.logger.info("Analyzing time series properties...")
        
        # Enhanced time series analysis
        analysis = {
            'data_overview': {
                'total_observations': len(self._time_series),
                'date_range': f"{self._time_series.index[0]} to {self._time_series.index[-1]}",
                'frequency': 'Monthly',
                'completeness': (1 - self._time_series.isna().sum() / len(self._time_series)) * 100
            }
        }
        
        # Stationarity test (only if statsmodels available)
        if STATSMODELS_AVAILABLE:
            adf_result = adfuller(self._time_series.dropna())
            analysis['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        else:
            analysis['stationarity'] = {'error': 'statsmodels not available'}
        
        # Seasonality detection
        if len(self._time_series) >= 24 and STATSMODELS_AVAILABLE:  # Need at least 2 years for seasonal decomposition
            try:
                decomposition = seasonal_decompose(
                    self._time_series.dropna(), 
                    model='additive', 
                    period=12
                )
                
                seasonal_strength = np.std(decomposition.seasonal) / np.std(self._time_series)
                trend_strength = np.std(decomposition.trend.dropna()) / np.std(self._time_series)
                
                analysis['seasonality'] = {
                    'seasonal_strength': seasonal_strength,
                    'trend_strength': trend_strength,
                    'has_seasonality': seasonal_strength > 0.1,
                    'has_trend': trend_strength > 0.1
                }
            except Exception as e:
                self.logger.warning(f"Seasonal decomposition failed: {e}")
                analysis['seasonality'] = {'error': str(e)}
        
        # Autocorrelation analysis
        if STATSMODELS_AVAILABLE:
            try:
                acf_values = acf(self._time_series.dropna(), nlags=min(40, len(self._time_series)//4), fft=False)
                pacf_values = pacf(self._time_series.dropna(), nlags=min(20, len(self._time_series)//8))
                
                analysis['autocorrelation'] = {
                    'acf_values': acf_values.tolist(),
                    'pacf_values': pacf_values.tolist(),
                    'significant_lags': [i for i, val in enumerate(acf_values[1:13], 1) if abs(val) > 0.2]
                }
            except Exception as e:
                self.logger.warning(f"Autocorrelation analysis failed: {e}")
                analysis['autocorrelation'] = {'error': str(e)}
        else:
            analysis['autocorrelation'] = {'error': 'statsmodels not available'}
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': float(self._time_series.mean()),
            'std': float(self._time_series.std()),
            'cv': float(self._time_series.std() / self._time_series.mean()),
            'trend_direction': 'increasing' if self._time_series.iloc[-12:].mean() > self._time_series.iloc[:12].mean() else 'decreasing'
        }
        
        self.logger.info("Time series analysis completed")
        return analysis
    
    def fit_sarima_model(
        self, 
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_select: bool = True
    ) -> Dict[str, Any]:
        """
        Fit SARIMA model to the time series.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            auto_select: Whether to automatically select best parameters
            
        Returns:
            Model fitting results
        """
        if not STATSMODELS_AVAILABLE:
            raise ForecastingError("SARIMA model requires statsmodels. Please install with: pip install statsmodels")
        
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        
        self.logger.info("Fitting SARIMA model...")
        
        # Auto parameter selection if requested
        if auto_select:
            order, seasonal_order = self._auto_select_sarima_params()
        
        # Default parameters if not provided
        if order is None:
            order = (1, 1, 1)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 12)
        
        try:
            # Fit SARIMA model
            model = ARIMA(
                self._time_series, 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit()
            
            # Store model
            self._fitted_models['sarima'] = fitted_model
            
            # Calculate performance metrics
            in_sample_forecast = fitted_model.fittedvalues
            performance = self._calculate_performance_metrics(
                self._time_series, 
                in_sample_forecast
            )
            self._model_performance['sarima'] = performance
            
            results = {
                'model_type': 'SARIMA',
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'performance': performance,
                'summary': str(fitted_model.summary())
            }
            
            self.logger.info(f"SARIMA model fitted successfully. AIC: {fitted_model.aic:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"SARIMA fitting failed: {e}")
            raise ForecastingError(f"SARIMA model fitting failed: {e}")
    
    def fit_ets_model(self, seasonal_periods: int = 12) -> Dict[str, Any]:
        """
        Fit Exponential Smoothing (ETS) model.
        
        Args:
            seasonal_periods: Number of periods in seasonal cycle
            
        Returns:
            Model fitting results
        """
        if not STATSMODELS_AVAILABLE:
            raise ForecastingError("ETS model requires statsmodels. Please install with: pip install statsmodels")
        
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        
        self.logger.info("Fitting ETS model...")
        
        try:
            # Try different ETS configurations
            configs = [
                {'error': 'add', 'trend': 'add', 'seasonal': 'add'},
                {'error': 'add', 'trend': 'add', 'seasonal': 'mul'},
                {'error': 'add', 'trend': None, 'seasonal': 'add'},
                {'error': 'add', 'trend': None, 'seasonal': None}
            ]
            
            best_model = None
            best_aic = float('inf')
            best_config = None
            
            for config in configs:
                try:
                    model = ETSModel(
                        self._time_series,
                        error=config['error'],
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=seasonal_periods if config['seasonal'] else None
                    )
                    
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_config = config
                        
                except Exception as e:
                    self.logger.debug(f"ETS config {config} failed: {e}")
                    continue
            
            if best_model is None:
                raise ForecastingError("No ETS configuration succeeded")
            
            # Store model
            self._fitted_models['ets'] = best_model
            
            # Calculate performance metrics
            in_sample_forecast = best_model.fittedvalues
            performance = self._calculate_performance_metrics(
                self._time_series, 
                in_sample_forecast
            )
            self._model_performance['ets'] = performance
            
            results = {
                'model_type': 'ETS',
                'configuration': best_config,
                'aic': best_model.aic,
                'bic': best_model.bic,
                'performance': performance,
                'summary': str(best_model.summary())
            }
            
            self.logger.info(f"ETS model fitted successfully. AIC: {best_model.aic:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"ETS fitting failed: {e}")
            raise ForecastingError(f"ETS model fitting failed: {e}")
    
    def fit_prophet_model(self) -> Dict[str, Any]:
        """
        Fit Prophet model (if available).
        
        Returns:
            Model fitting results
        """
        if not PROPHET_AVAILABLE:
            raise ForecastingError("Prophet is not available. Install with: pip install prophet")
        
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        
        self.logger.info("Fitting Prophet model...")
        
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': self._time_series.index,
                'y': self._time_series.values
            })
            
            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Fit model
            fitted_model = model.fit(prophet_data)
            
            # Store model
            self._fitted_models['prophet'] = fitted_model
            
            # Generate in-sample predictions for evaluation
            in_sample_pred = fitted_model.predict(prophet_data)
            in_sample_forecast = pd.Series(
                in_sample_pred['yhat'].values,
                index=self._time_series.index
            )
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(
                self._time_series, 
                in_sample_forecast
            )
            self._model_performance['prophet'] = performance
            
            results = {
                'model_type': 'Prophet',
                'performance': performance,
                'components': ['trend', 'yearly', 'seasonal'],
                'changepoints': len(fitted_model.changepoints)
            }
            
            self.logger.info(f"Prophet model fitted successfully. MAPE: {performance['mape']:.2f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"Prophet fitting failed: {e}")
            raise ForecastingError(f"Prophet model fitting failed: {e}")
    
    def forecast(
        self, 
        steps: Optional[int] = None,
        model_name: str = 'auto',
        include_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecasts using specified or best model.
        
        Args:
            steps: Number of periods to forecast
            model_name: Model to use ('sarima', 'ets', 'prophet', 'auto')
            include_intervals: Whether to include prediction intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Forecast results
        """
        if steps is None:
            steps = self.config.get('forecast.horizon', 12)
        
        # Select best model if auto
        if model_name == 'auto':
            model_name = self._select_best_model()
        
        if model_name not in self._fitted_models:
            raise ValueError(f"Model '{model_name}' not fitted. Call fit_{model_name}_model() first.")
        
        self.logger.info(f"Generating {steps}-step forecast using {model_name} model...")
        
        fitted_model = self._fitted_models[model_name]
        
        try:
            if model_name == 'sarima':
                if include_intervals:
                    try:
                        res = fitted_model.get_forecast(steps=steps)
                        ci = res.conf_int(alpha=(1 - confidence_level))
                        forecast_values = res.predicted_mean
                        lower_bound = ci.iloc[:, 0].values if hasattr(ci, 'iloc') else ci[:, 0]
                        upper_bound = ci.iloc[:, 1].values if hasattr(ci, 'iloc') else ci[:, 1]
                    except Exception:
                        # Fallback: no intervals
                        forecast_values = fitted_model.forecast(steps=steps)
                        lower_bound = None
                        upper_bound = None
                else:
                    forecast_values = fitted_model.forecast(steps=steps)
                    lower_bound = None
                    upper_bound = None
                    
            elif model_name == 'ets':
                if include_intervals and hasattr(fitted_model, 'get_prediction'):
                    try:
                        res = fitted_model.get_prediction(start=len(self._time_series), end=len(self._time_series)+steps-1)
                        ci = res.conf_int(alpha=(1 - confidence_level))
                        forecast_values = res.predicted_mean
                        lower_bound = ci.iloc[:, 0].values if hasattr(ci, 'iloc') else ci[:, 0]
                        upper_bound = ci.iloc[:, 1].values if hasattr(ci, 'iloc') else ci[:, 1]
                    except Exception:
                        # Fallback to point forecast only
                        forecast_values = fitted_model.forecast(steps=steps)
                        lower_bound = None
                        upper_bound = None
                else:
                    forecast_values = fitted_model.forecast(steps=steps)
                    lower_bound = None
                    upper_bound = None
                    
            elif model_name == 'prophet':
                # Create future dataframe
                future_dates = pd.date_range(
                    start=self._time_series.index[-1] + pd.DateOffset(months=1),
                    periods=steps,
                    freq='MS'
                )
                
                future_df = pd.DataFrame({'ds': future_dates})
                forecast_result = fitted_model.predict(future_df)
                
                forecast_values = forecast_result['yhat'].values
                if include_intervals:
                    lower_bound = forecast_result['yhat_lower'].values
                    upper_bound = forecast_result['yhat_upper'].values
                else:
                    lower_bound = None
                    upper_bound = None
            
            # Create forecast index
            forecast_index = pd.date_range(
                start=self._time_series.index[-1] + pd.DateOffset(months=1),
                periods=steps,
                freq='MS'
            )
            
            # Comprehensive forecast validation and fallback
            forecast_valid = False
            
            # Check if forecast_values is valid
            if forecast_values is None:
                self.logger.warning("SARIMA produced None forecast, using simple trend forecast")
                forecast_values = self._simple_trend_forecast(steps)
                forecast_valid = True
            elif isinstance(forecast_values, (int, float)):
                if np.isnan(forecast_values) or np.isinf(forecast_values):
                    self.logger.warning("SARIMA produced invalid scalar forecast, using simple trend forecast")
                    forecast_values = self._simple_trend_forecast(steps)
                    forecast_valid = True
                else:
                    # Convert single value to array
                    forecast_values = np.full(steps, forecast_values)
                    forecast_valid = True
            elif hasattr(forecast_values, '__iter__'):
                forecast_array = np.array(forecast_values)
                if len(forecast_array) == 0:
                    self.logger.warning("SARIMA produced empty forecast, using simple trend forecast")
                    forecast_values = self._simple_trend_forecast(steps)
                    forecast_valid = True
                elif np.all(np.isnan(forecast_array)) or np.all(np.isinf(forecast_array)):
                    self.logger.warning("SARIMA produced all NaN/Inf forecast, using simple trend forecast")
                    forecast_values = self._simple_trend_forecast(steps)
                    forecast_valid = True
                elif np.any(np.isnan(forecast_array)) or np.any(np.isinf(forecast_array)):
                    self.logger.warning("SARIMA produced partial NaN/Inf forecast, using simple trend forecast")
                    forecast_values = self._simple_trend_forecast(steps)
                    forecast_valid = True
                else:
                    forecast_values = forecast_array
                    forecast_valid = True
            else:
                self.logger.warning("SARIMA produced unknown forecast type, using simple trend forecast")
                forecast_values = self._simple_trend_forecast(steps)
                forecast_valid = True
            
            if not forecast_valid:
                self.logger.error("All forecast validation failed, using emergency fallback")
                forecast_values = self._simple_trend_forecast(steps)
            
            # Compile results
            forecast_data = {
                'model_used': model_name,
                'forecast_values': pd.Series(forecast_values, index=forecast_index),
                'confidence_level': confidence_level if include_intervals else None,
                'lower_bound': pd.Series(lower_bound, index=forecast_index) if lower_bound is not None else None,
                'upper_bound': pd.Series(upper_bound, index=forecast_index) if upper_bound is not None else None,
                'model_performance': self._model_performance.get(model_name, {}),
                'forecast_metadata': {
                    'forecast_date': datetime.now().isoformat(),
                    'data_end_date': self._time_series.index[-1].isoformat(),
                    'forecast_horizon': steps
                }
            }
            
            self.logger.info(f"Forecast completed successfully using {model_name} model")
            return forecast_data
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            raise ForecastingError(f"Forecasting failed: {e}")
    
    def validate_models(self, test_periods: int = 6) -> Dict[str, Dict[str, float]]:
        """
        Validate fitted models using time series cross-validation.
        
        Args:
            test_periods: Number of periods to use for validation
            
        Returns:
            Validation results for all fitted models
        """
        if self._time_series is None:
            raise ValueError("No time series loaded. Call load_data() first.")
        
        self.logger.info(f"Validating models using {test_periods}-period holdout...")
        
        # Split data
        train_size = len(self._time_series) - test_periods
        train_data = self._time_series.iloc[:train_size]
        test_data = self._time_series.iloc[train_size:]
        
        validation_results = {}
        
        for model_name in self._fitted_models.keys():
            try:
                self.logger.info(f"Validating {model_name} model...")
                
                # Refit model on training data only
                if model_name == 'sarima':
                    model = ARIMA(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    fitted = model.fit()
                    forecast_result = fitted.forecast(steps=test_periods)
                    predictions = forecast_result if isinstance(forecast_result, np.ndarray) else forecast_result[0]
                    
                elif model_name == 'ets':
                    model = ETSModel(train_data, seasonal_periods=12)
                    fitted = model.fit()
                    forecast_result = fitted.forecast(steps=test_periods)
                    predictions = forecast_result if isinstance(forecast_result, np.ndarray) else forecast_result[0]
                    
                elif model_name == 'prophet':
                    if not PROPHET_AVAILABLE:
                        continue
                    
                    prophet_train = pd.DataFrame({
                        'ds': train_data.index,
                        'y': train_data.values
                    })
                    
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    fitted = model.fit(prophet_train)
                    
                    future_df = pd.DataFrame({'ds': test_data.index})
                    forecast_result = fitted.predict(future_df)
                    predictions = forecast_result['yhat'].values
                
                # Calculate validation metrics
                validation_metrics = self._calculate_performance_metrics(test_data, predictions)
                validation_results[model_name] = validation_metrics
                
                self.logger.info(f"{model_name} validation - MAPE: {validation_metrics['mape']:.2f}%")
                
            except Exception as e:
                self.logger.warning(f"Validation failed for {model_name}: {e}")
                validation_results[model_name] = {'error': str(e)}
        
        return validation_results
    
    def _auto_select_sarima_params(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Fast SARIMA parameter selection with reduced grid search.
        
        Returns:
            Tuple of (order, seasonal_order)
        """
        self.logger.info("Auto-selecting SARIMA parameters (fast mode)...")
        
        # Minimal grid search for speed
        param_combinations = [
            ((1, 1, 1), (1, 1, 1, 12)),
            ((2, 1, 2), (1, 1, 1, 12)),
            ((1, 1, 0), (0, 1, 1, 12)),
            ((0, 1, 1), (1, 1, 0, 12))
        ]
        
        best_aic = float('inf')
        best_params = ((1, 1, 1), (1, 1, 1, 12))
        
        for order, seasonal_order in param_combinations:
            try:
                model = ARIMA(
                    self._time_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(maxiter=50)  # Limit iterations
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (order, seasonal_order)
                    
            except Exception:
                continue
        
        self.logger.info(f"Best SARIMA parameters: {best_params[0]}, seasonal: {best_params[1]}")
        return best_params
    
    def _select_best_model(self) -> str:
        """
        Select best model based on performance metrics.
        
        Returns:
            Name of best model
        """
        if not self._model_performance:
            raise ValueError("No models fitted for comparison")
        
        # Use MAPE as primary criterion
        best_model = min(
            self._model_performance.keys(),
            key=lambda x: self._model_performance[x].get('mape', float('inf'))
        )
        
        self.logger.info(f"Best model selected: {best_model}")
        return best_model
    
    def _calculate_performance_metrics(self, actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Align data
        min_length = min(len(actual), len(predicted))
        actual_aligned = actual.iloc[-min_length:].values
        predicted_aligned = predicted[-min_length:]
        
        # Remove any NaN values
        mask = ~(np.isnan(actual_aligned) | np.isnan(predicted_aligned))
        actual_clean = actual_aligned[mask]
        predicted_clean = predicted_aligned[mask]
        
        if len(actual_clean) == 0:
            return {'error': 'No valid data points for evaluation'}
        
        # Use sklearn if available, otherwise use fallback functions
        if SKLEARN_AVAILABLE:
            mae_val = float(mean_absolute_error(actual_clean, predicted_clean))
            mse_val = float(mean_squared_error(actual_clean, predicted_clean))
            mape_val = float(mean_absolute_percentage_error(actual_clean, predicted_clean) * 100)
        else:
            mae_val = float(_fallback_mean_absolute_error(actual_clean, predicted_clean))
            mse_val = float(_fallback_mean_squared_error(actual_clean, predicted_clean))
            mape_val = float(_fallback_mean_absolute_percentage_error(actual_clean, predicted_clean))
        
        metrics = {
            'mae': mae_val,
            'mse': mse_val,
            'rmse': float(np.sqrt(mse_val)),
            'mape': mape_val,
            'r_squared': float(np.corrcoef(actual_clean, predicted_clean)[0, 1] ** 2) if len(actual_clean) > 1 else 0
        }
        
        return metrics
    
    def save_models(self, file_path: Optional[str] = None) -> str:
        """
        Save fitted models to file.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = self.config.get_output_path('models/fitted_models.pkl')
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'fitted_models': self._fitted_models,
            'model_performance': self._model_performance,
            'metadata': {
                'save_date': datetime.now().isoformat(),
                'config': dict(self.config._config)
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Models saved to: {file_path}")
        return file_path
    
    def load_models(self, file_path: str) -> None:
        """
        Load fitted models from file.
        
        Args:
            file_path: Path to saved models
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self._fitted_models = model_data['fitted_models']
        self._model_performance = model_data['model_performance']
        
        self.logger.info(f"Models loaded from: {file_path}")
        self.logger.info(f"Available models: {list(self._fitted_models.keys())}")
    
    def _simple_trend_forecast(self, steps: int) -> np.ndarray:
        """
        Simple trend-based forecast fallback when SARIMA fails.
        
        Args:
            steps: Number of periods to forecast
            
        Returns:
            Forecast values array
        """
        if self._time_series is None or len(self._time_series) < 2:
            # Ultimate fallback: use reasonable default values
            base_value = 3000000  # 3M TRY monthly sales
            growth_rate = 0.02    # 2% monthly growth
            return base_value * (1 + growth_rate) ** np.arange(steps)
        
        # Calculate simple trend from last 12 months (or available data)
        recent_data = self._time_series.dropna().tail(min(12, len(self._time_series)))
        
        if len(recent_data) < 2:
            # Use last value with small growth
            last_value = recent_data.iloc[-1] if len(recent_data) > 0 else 3000000
            growth_rate = 0.02
            return last_value * (1 + growth_rate) ** np.arange(steps)
        
        # Linear trend calculation
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Simple linear regression
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
        intercept = y.mean() - slope * x.mean()
        
        # Project forward
        future_x = np.arange(len(recent_data), len(recent_data) + steps)
        forecast_values = intercept + slope * future_x
        
        # Ensure positive values
        forecast_values = np.maximum(forecast_values, recent_data.iloc[-1] * 0.5)
        
        self.logger.info(f"Simple trend forecast: {forecast_values[0]:,.0f} to {forecast_values[-1]:,.0f} TRY")
        return forecast_values
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available fitted models."""
        return list(self._fitted_models.keys())
    
    @property
    def model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get model performance metrics."""
        return self._model_performance.copy()