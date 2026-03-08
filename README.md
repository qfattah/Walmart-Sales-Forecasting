# Walmart Sales Forecasting

## Project Overview

This project develops a time series forecasting framework to predict weekly sales for Walmart stores and departments using historical sales data, store information, holiday events, and economic indicators.

The objective is to evaluate multiple forecasting approaches and determine which model produces the most accurate and stable predictions across different product categories.

The project integrates:

- Time series visualization and exploratory analysis  
- Data preprocessing and missing value handling  
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test  
- SARIMA time series modeling  
- Exponential Smoothing (Holt-Winters) forecasting  
- Facebook Prophet forecasting  
- Model evaluation using MAE, MSE, and MAPE  

Competition link:  
https://www.kaggle.com/competitions/walmart-sales-prediction/overview  

You can review the code here:  
https://github.com/qfattah/Walmart-Sales-Forecasting/blob/main/Walmart_Sales_Forecat.ipynb

---

## Business Context

Retail forecasting plays a critical role in:

- Inventory planning  
- Demand forecasting  
- Workforce scheduling  
- Promotion planning  
- Supply chain optimization  

Retail sales data often exhibits **strong seasonality**, **holiday effects**, and **non-linear trends**, making time series forecasting techniques essential for accurate demand prediction.

The goal of this project is to evaluate different forecasting models and identify the most reliable method for predicting sales across product categories.

---

## Technical Approach

### 1. Data Import and Visualization

The dataset is loaded and visualized to understand:

- Overall sales trends
- Seasonal patterns
- Category-level demand variation

Sales data is analyzed separately for three product categories:

- **Men Clothing**
- **Women Clothing**
- **Other Clothing**

Initial visualization helps identify seasonal behavior and overall sales magnitude.

---

### 2. Data Preprocessing

Several preprocessing steps are applied before modeling:

- Splitting the dataset by product category
- Handling missing values using interpolation
- Ensuring the time series structure is consistent
- Preparing the data for statistical modeling

Proper preprocessing ensures model stability and improves forecast reliability.

---

### 3. Stationarity Testing (ADF Test)

A custom function `test_stationarity()` is implemented using the **Augmented Dickey-Fuller (ADF) test** to determine whether each time series is stationary.

The function outputs:

- ADF statistic  
- p-value  
- Number of lags used  
- Number of observations used  
- Critical values at 1%, 5%, and 10% significance levels  

#### Results

| Dataset | Result |
|------|------|
| Women Clothing | Stationary (p < 0.05) |
| Men Clothing | Non-stationary |
| Other Clothing | Non-stationary |

#### Action Taken

First-order differencing was applied to the non-stationary series (`df_men`, `df_other`) before model training.

---

## Forecasting Models

### 1. SARIMA Model

SARIMA extends ARIMA by incorporating seasonal components.

The SARIMA workflow includes:

- Splitting the dataset into training and testing sets
- Fitting the SARIMA model on stationary data
- Generating forecasts
- Inverting differencing to return forecasts to the original scale
- Evaluating performance using MAE, MSE, and MAPE

#### Model Configuration

Order (p,d,q) = (1,0,1)  
Seasonal Order (P,D,Q,s) = (1,1,0,12)

#### Model Selection Process

ACF and PACF plots were used to determine model parameters.

- ACF showed strong spikes at lags **1, 2, and 11**
- PACF showed strong spikes at **1 and 11**

This indicated both **short-term autocorrelation and yearly seasonal effects**.

---

### 2. Exponential Smoothing (Holt-Winters)

Holt-Winters Exponential Smoothing models level, trend, and seasonality simultaneously.

#### Model Configuration

trend = 'add'  
seasonal = 'add'  
seasonal_periods = 12

#### Workflow

- Train/test split (80/20)
- Fit model to training data
- Generate forecasts for the validation period
- Compare forecasts against actual values

#### Evaluation Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Exponential smoothing produced **smooth and interpretable forecasts** that captured both trend and seasonality.

---

### 3. Prophet Forecasting

Prophet is a forecasting framework developed by Meta (Facebook) designed for business time series data.

It decomposes time series into:

- Trend
- Seasonality
- Holiday effects

#### Workflow

- Convert dataset into Prophet format (`ds`, `y`)
- Fit model on training data
- Generate future forecasts
- Compare predictions against actual values

#### Key Advantages

- Automatic changepoint detection
- Handles missing data
- Captures non-linear trends
- Built-in seasonal components

---

## Model Evaluation

Models were evaluated using the following metrics:

- **MAE** — Mean Absolute Error  
- **MSE** — Mean Squared Error  
- **MAPE** — Mean Absolute Percentage Error  

| Category | Best Model | MAPE | MAE | MSE |
|------|------|------|------|------|
| Men Clothing | Exponential Smoothing | 3.62% | 25.70 | 778 |
| Women Clothing | Exponential Smoothing | 3.78% | 120.97 | 28,184 |
| Other Clothing | Exponential Smoothing | 5.11% | 62.89 | 6,312 |

### Key Finding

**Exponential Smoothing consistently outperformed SARIMA and Prophet across all categories.**

It produced the lowest error metrics while requiring fewer modeling assumptions and less parameter tuning.

---

## Key Skills Demonstrated

- Time Series Forecasting
- Stationarity Testing (ADF)
- SARIMA Modeling
- Exponential Smoothing (Holt-Winters)
- Prophet Forecasting
- Model Evaluation and Comparison
- Data Preprocessing and Visualization

---

## Tools & Libraries

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Statsmodels  
- Prophet  
- Scikit-learn  

---

## Impact & Practical Applications

This project demonstrates how different forecasting techniques perform when applied to retail sales data with strong seasonal patterns.

The methodology can be extended to:

- Inventory demand forecasting
- Supply chain planning
- Retail promotion forecasting
- Category-level demand prediction
- Store-level operational planning

---

## Author

Qusai Fattah  
Data Science | Time Series Forecasting | Supply Chain Analytics
