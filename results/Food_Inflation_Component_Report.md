# ESI Report Component: Food Inflation

## Dataset
- **Indicator**: India Food Inflation
- **Period**: 1991–2024
- **Source**: RBI / MOSPI / World Bank

## Model
- **Algorithm**: ARIMA(p,d,q)
- **Selection Method**: Hyperparameters selected via Auto-ARIMA

## Evaluation
- **Metrics**: 
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- **Backtest Period**: 2019–2024

## Visualizations
- **Actual vs Forecast**: Demonstrates model fit and accuracy over the testing period.
- **Residual Plot**: Verifies that residuals resemble random noise with no visible patterns.
