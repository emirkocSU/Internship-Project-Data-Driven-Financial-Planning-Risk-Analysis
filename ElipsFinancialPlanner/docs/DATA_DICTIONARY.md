# Data Dictionary

## sales_history.cleaned.csv

- date: Month start date (YYYY-MM-01), timezone-naive datetime
- sales_total_try: Total sales in TRY for the month (float64), >= 0
- opex_try: Operating expenses in TRY for the month (float64), >= 0
- usdtry: Average FX USD/TRY for the month (float64), > 0, constrained to [3.5, max_fx_rate]
- is_holiday_peak (optional): Indicator (0/1) for Nov–Dec
- promo_intensity (optional): 0–1 synthetic promotional signal
- price_change_flag (optional): 0/1, large FX monthly change
- stockout_flag (optional): 0/1, rare stockout proxy
- avg_price_try (optional): Derived average unit price in TRY
- units_sold (optional): Derived units sold (sales_total_try / avg_price_try)

### Allowed Ranges
- sales_total_try: [0, +inf)
- opex_try: [0, +inf)
- usdtry: (0, max_fx_rate] (config.data.max_fx_rate)

### Integrity Report (excerpt)

```
{'rows': 1200, 'date_range': ['2018-01-02 00:00:00', '2025-09-28 00:00:00'], 'schema': {'date': 'datetime64[ns]', 'sales_total_try': 'float64', 'opex_try': 'float64', 'usdtry': 'float64'}, 'min': {'sales_total_try': 1621765.3054177188, 'opex_try': 470856.6031057985, 'usdtry': 3.95}, 'max': {'sales_total_try': 4321111.111111111, 'opex_try': 1404529.1077610904, 'usdtry': 29.6}, 'nan_counts': {'date': 0, 'sales_total_try': 20, 'opex_try': 20, 'usdtry': 0, 'is_holiday_peak': 0, 'promo_intensity': 0, 'price_change_flag': 698, 'stockout_flag': 698, 'avg_price_try': 0, 'units_sold': 20, 'cogs_try': 20, 'gross_margin_try': 20, 'competitor_index': 0, 'data_quality_flag': 0, 'imputation_method': 0, 'is_black_friday': 0, 'is_ramadan': 0, 'is_back_to_school': 0}}
```
