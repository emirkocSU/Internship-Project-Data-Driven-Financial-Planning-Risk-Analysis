# Elips Medikal Financial Planning System - Focused Senior Implementation Plan

## 🎯 OBJECTIVE
Build a focused, high-quality Python CLI financial planning system for Elips Medikal with core forecasting, financial planning, and risk analysis capabilities.

---

## PHASE 1: PROJECT FOUNDATION (Week 1)

### 1.1 Simplified Project Structure
- Create main directory: `ElipsFinancialPlanner/`
- Create essential subdirectories: `src/`, `config/`, `data/`, `outputs/`, `tests/`
- Initialize Python package with `__init__.py` files
- Create `pyproject.toml` for modern Python packaging
- Create `.gitignore` for Python projects
- Create `README.md` with basic project info

### 1.2 Core Dependencies
- Create `requirements.txt` with essential packages:
  - pandas>=1.3.0
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - statsmodels>=0.13.0
  - click>=8.0.0
  - pyyaml>=6.0
  - prophet>=1.1.0 (optional)
- Setup virtual environment automation script

### 1.3 Single Configuration System
- Create `config/settings.yaml` with all parameters:
  ```yaml
  random_seed: 42
  forecast:
    model: "sarima"  # sarima, ets, prophet
    horizon: 12
    seasonal: true
  finance:
    tax_rate: 0.22
    cogs_share: 0.65
    beta_fx: 0.15
    dso: 45  # Days Sales Outstanding
    dpo: 30  # Days Payable Outstanding
    dio: 60  # Days Inventory Outstanding
    depreciation: 50000  # monthly TRY
    capex: 100000  # monthly TRY
  scenarios:
    best: {sales_change: 0.15, fx_change: 0.0, opex_change: 0.0}
    fx_shock: {sales_change: -0.05, fx_change: 0.25, opex_change: 0.0}
    cost_cut: {sales_change: 0.0, fx_change: 0.0, opex_change: -0.10}
  risk:
    mc_runs: 1000
    sales_vol: 0.15
    fx_vol: 0.20
  ```
- Create `src/config.py` to load and validate configuration

### 1.4 Basic Logging
- Setup standard Python logging with rotating file handler
- Create simple error handling with clear messages
- Add progress indicators for long-running operations

---

## PHASE 2: DATA FOUNDATION & MODELING (Week 2)

### 2.1 Simple Synthetic Dataset
- Create `data/sales_history.csv` with columns:
  - date (YYYY-MM-DD)
  - sales_total_try
  - opex_try
  - usdtry
- Generate 5-7 years of monthly data (2018-2024)
- Include: trend + seasonality + tender spikes + FX correlation + opex inflation
- Simulate major shocks with parameter-driven approach (no complex rules)

### 2.2 Data Loading & Validation
- Create `src/data_loader.py` with pandas CSV reading
- Implement basic validation with Pandera:
  - Date format and continuity
  - Positive values for sales/opex
  - Reasonable FX range
- Handle missing values with forward fill
- Add data quality summary report

### 2.3 Streamlined EDA
- Create `src/eda.py` with essential analysis:
  - Time series plots (trend/seasonality)
  - YoY growth rate calculation
  - FX correlation analysis
  - Basic summary statistics
- Generate 2-3 key visualization plots
- Save EDA results to outputs/

### 2.4 Core Forecasting Models
- Create `src/forecasting.py` with:
  - **SARIMA**: Primary model with auto parameter selection
  - **ETS**: Backup exponential smoothing model
  - **Prophet**: Optional with flag-based activation
- Implement TimeSeriesSplit cross-validation (3-4 folds)
- Calculate MAPE, SMAPE, and prediction interval coverage
- Save fitted models and validation results

---

## PHASE 3: FINANCIAL PLANNING ENGINE (Week 3)

### 3.1 Core Financial Calculations
- Create `src/financial_planner.py` with formulas:
  ```python
  # Core calculations
  COGS = cogs_share * Sales * (1 + beta_fx * fx_change)
  Gross_Profit = Sales - COGS
  EBIT = Gross_Profit - OpEx
  Tax = EBIT * tax_rate (if EBIT > 0)
  NOPAT = EBIT - Tax
  
  # Working Capital
  AR = Sales * DSO / 30
  AP = COGS * DPO / 30
  INV = COGS * DIO / 30
  Delta_WC = (AR + INV - AP)_current - (AR + INV - AP)_previous
  
  # Cash Flow
  NetCash = NOPAT + Depreciation - Capex - Delta_WC
  ```

### 3.2 Scenario Application Engine
- Create `src/scenarios.py` with 4 core scenarios:
  - **Base**: Direct forecast results
  - **Best**: Sales +15%, FX stable, OpEx stable
  - **FX Shock**: Sales -5%, FX +25%, OpEx stable
  - **Cost Cut**: Sales stable, FX stable, OpEx -10%
- Implement parameter override system
- Generate scenario comparison outputs

### 3.3 Financial Planning Pipeline
- Integrate forecasting with financial calculations
- Generate complete financial projections for 12 months
- Create working capital and cash flow projections
- Output standardized results format

---

## PHASE 4: RISK ANALYSIS & CLI (Week 4)

### 4.1 Monte Carlo Risk Analysis
- Create `src/risk_analysis.py` with:
  - Sales volatility simulation (normal distribution)
  - FX rate volatility simulation
  - 1,000-5,000 Monte Carlo runs
  - Calculate VaR95 and CVaR95 for NetCash or annual EBIT
- Generate risk distribution plots
- Create risk summary report

### 4.2 CLI Interface
- Create `src/cli.py` with Click framework
- Implement core commands:
  - `fit`: Train forecasting models
  - `forecast`: Generate base forecast
  - `plan`: Create financial projections
  - `simulate`: Run risk analysis
  - `compare`: Optional scenario comparison
- Add progress bars and status messages
- Implement clear error handling

### 4.3 Output Generation
- Create `src/output_generator.py` for standardized outputs:
  - **CSV**: Financial projections with standard columns
    (date, sales, cogs, gross, opex, ebit, tax, nopat, ar, ap, inv, delta_wc, netcash)
  - **PNG Charts**: 2-3 key visualizations
    - Historical vs forecast line chart
    - Scenario comparison bar chart
    - Risk distribution (fan chart)
- Simple, clean chart styling

### 4.4 Testing & Validation
- Create unit tests for financial formulas
- Test scenario override functionality
- Validate forecast output shapes and NaN handling
- Add end-to-end pipeline test
- Ensure reproducibility with random seed

---

## TECHNICAL SPECIFICATIONS

### Performance Requirements
- **Speed**: Full analysis <5 seconds for 5-7 years monthly data
- **Memory**: Minimal memory footprint
- **Accuracy**: MAPE ≤12-15%, Prediction Interval coverage 80-95%
- **Risk**: VaR95/CVaR95 reporting for NetCash

### File Structure
```
ElipsFinancialPlanner/
├── pyproject.toml          # Package configuration
├── README.md               # User documentation
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration loading
│   ├── data_loader.py      # Data loading & validation
│   ├── eda.py             # Exploratory analysis
│   ├── forecasting.py     # SARIMA/ETS/Prophet models
│   ├── financial_planner.py # Financial calculations
│   ├── scenarios.py       # Scenario analysis
│   ├── risk_analysis.py   # Monte Carlo VaR/CVaR
│   ├── output_generator.py # CSV/PNG outputs
│   └── cli.py             # Command line interface
├── config/                 # Configuration files
│   └── settings.yaml      # All parameters
├── data/                   # Data files
│   └── sales_history.csv  # Historical data
├── outputs/               # Generated outputs
├── tests/                 # Test files
└── main.py               # Entry point
```

### Data Schema
```csv
# data/sales_history.csv
date,sales_total_try,opex_try,usdtry
2018-01-01,1500000,400000,3.78
2018-02-01,1450000,410000,3.82
...
```

### Configuration Parameters
- All "magic numbers" moved to YAML
- Random seed enforced for reproducibility
- Financial ratios configurable
- Scenario parameters adjustable
- Risk analysis parameters tunable

---

## SUCCESS CRITERIA

### ✅ Final Acceptance Criteria:
1. **Accuracy**: MAPE ≤12-15% on validation set
2. **Coverage**: Prediction intervals cover 80-95% of actual values
3. **Risk Reporting**: VaR95/CVaR95 calculated and reported
4. **Performance**: Full pipeline execution <5 seconds
5. **Outputs**: Complete CSV with all financial metrics + 2-3 PNG charts
6. **Reproducibility**: Consistent results with same random seed
7. **Flexibility**: All key parameters configurable via YAML

### 🔧 Quality Standards:
- Clean, readable code with basic error handling
- Unit tests for core financial calculations
- Clear documentation and examples
- Proper logging and progress feedback
- Standardized output formats

### 📊 Core Deliverables:
1. **Forecasting**: 12-month sales projections with confidence intervals
2. **Financial Planning**: Complete income statement and cash flow projections
3. **Scenario Analysis**: 4 core scenarios with comparison
4. **Risk Analysis**: Monte Carlo VaR/CVaR analysis
5. **CLI Tool**: Professional command-line interface
6. **Documentation**: Clear setup and usage instructions

---

## 4-WEEK SPRINT PLAN

### Week 1: Foundation
- Project setup + dependencies
- Configuration system
- Data generation + loading
- Basic logging

### Week 2: Core Analytics
- EDA implementation
- Forecasting models (SARIMA/ETS)
- Model validation
- Prophet integration (optional)

### Week 3: Financial Engine
- Financial calculations
- Scenario system
- Working capital modeling
- Cash flow projections

### Week 4: Risk & Interface
- Monte Carlo analysis
- CLI implementation
- Output generation
- Testing + documentation

**🎯 RESULT: FOCUSED, HIGH-QUALITY FINANCIAL PLANNING SYSTEM** 