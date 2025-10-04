## Walmart Sales Forecasting

This project aims to build a machine learning model to predict weekly sales for Walmart stores and departments using historical sales data, store information, holiday events, and economic factors. The project involves data preprocessing, feature engineering, and model training to improve sales forecasting and handle seasonal trends. The goal is to create an accurate and robust predictive model evaluated using RMSE. 

Competition link: [Walmart Sales Prediction on Kaggle](https://www.kaggle.com/competitions/walmart-sales-prediction/overview)  
You can review the code here: [Walmart_Sales_Forecat](https://github.com/qfattah/Walmart-Sales-Forecasting/blob/main/Walmart_Sales_Forecat.ipynb)


## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Libraries](#libraries)
- [Walmart Sales Forecasting Code Sections](#walmart-sales-forecasting-code-sections)
  - [Importing the Data and Visualizing it](#importing-the-data-and-visualizing-it)
  - [Data Pre-Processing](#data-pre-processing)
  - [Stationarity Testing (ADF Test)](#stationarity-testing-adf-test)
- [SARIMA Model](#sarima-model)
- [Exponential Smoothing Modeling](#️-exponential-smoothing-modeling)
- [Prophet Forecasting](#-prophet-forecasting)
- [Conclusion - Model Selection Note](#conclusion---model-selection-note)



## Introduction

The project involves the following steps:
1. **Importing Libraries**: Importing necessary libraries for forecasting, data processing, and visualization.
2. **Importing the Data and Visualizing it**: Observe trends, seaonal trends and sales magnitude.
3. **Data Pre-Processing**: Handle missing values, and preprocessing the data eo ensure stationarity.
4. **SARIMA Model Training and Forecasting**: Univariate time series forecasting model that extends ARIMA by incorporating seasonal components. Combines autoregressive (AR), differencing (I), and moving average (MA) terms with seasonal analogs to capture both short-term dependencies and repeating seasonal patterns in the data.
5. **Exponential Smoothing Model Training and Forecasting**: Univariate forecasting technique that applies weighted averages of past observations, where the weights decrease exponentially over time. Suitable for capturing level, trend, and seasonality components in time series data depending on the chosen variant (Simple, Holt, Holt-Winters).
6. **Prophet Model Forecasting**: Univariate forecasting model developed by Facebook that decomposes time series into trend, seasonality, and holiday effects. Designed for business time series with strong seasonal patterns and missing data, it provides interpretable parameters and handles non-linear trends with changepoint detection.


## Setup

### Prerequisites
- Python 3.x
- Jupyter Notebook

### Libraries
Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn statsmodels tqdm scikit-learn prophet tensorflow statsmodels prophet tensorflow;
```

## Walmart Sales Forecasting Code Sections

### Importing the Data and Visualizing it
- Visualized the Sales over time by Category.

### Data Pre-Processing
- Split the dataset by product category and applied interpolation to address missing values.

### Stationarity Testing (ADF Test)
- A custom function `test_stationarity()` was implemented to assess time series stationarity using the **Augmented Dickey-Fuller (ADF) test**.
- The function outputs key diagnostic information:
  - ADF statistic  
  - p-value  
  - Number of lags used  
  - Number of observations used  
  - Critical values at 1%, 5%, and 10% confidence levels  
  - A conclusion on stationarity based on the p-value

- The test was applied to the following product category datasets:
  - `df_men` — *Men Clothing Sales*
  - `df_women` — *Women Clothing Sales*
  - `df_other` — *Other Clothing Sales*

- **Results:**
  - `df_women` was **stationary** (*p < 0.05*)  
  - `df_men` and `df_other` were **not stationary** (*p > 0.05*)

- **Action Taken:**
  - Applied **first-order differencing** to the non-stationary time series (`df_men` and `df_other`) to achieve stationarity prior to model training.

## SARIMA Model

- Defined a function `run_sarima_model()` to train and evaluate a SARIMA model for each product category.

- **Workflow includes:**
  - Splitting the time series into training and testing sets (default 80/20).
  - Fitting the **SARIMA** model on the differenced series to ensure stationarity.
  - Forecasting on the test set and **inverting the differencing** to return forecasts to the original scale.
  - Calculating evaluation metrics:  
    - **MAE** (Mean Absolute Error)  
    - **MSE** (Mean Squared Error)  
    - **MAPE** (Mean Absolute Percentage Error)

- **Visualization:**
  - Overlays the forecast on top of the original series with a train-test split marker.

- **ACF/PACF Analysis:**
  - ACF showed significant spikes at lags 1, 2, and 11 → suggests autocorrelation and seasonality.
  - PACF showed strong lags at 1 and 11 → supports inclusion of AR and seasonal AR terms.
  - Seasonal differencing at lag 12 used to capture yearly seasonality in monthly data.

- **Model Parameters:**
  - **Order (p,d,q)** = (1, 0, 1)  
  - **Seasonal Order (P,D,Q,s)** = (1, 1, 0, 12)  
    - Chosen based on ACF/PACF plots and constrained by limited data (48 months).

- **Results:**
  - Forecasts were generated for:
    - `df_men` — *Men Clothing*
    - `df_women` — *Women Clothing*
    - `df_other` — *Other Clothing*
  - Forecasts were evaluated and visualized for each category.

- **Note:**  
  - Seasonal MA (`Q`) was initially suggested by ACF but set to 0 due to the limited training window and improved performance.


### 📈 Exponential Smoothing Modeling

- A function `run_exponential_smoothing()` was implemented to forecast sales using **Holt-Winters Exponential Smoothing**.

- **Workflow includes:**
  - Splitting the dataset into training and testing sets (default: 80/20 split).
  - Fitting an **additive trend and seasonal** Exponential Smoothing model (`trend='add'`, `seasonal='add'`, `seasonal_periods=12`).
  - Forecasting on the test set without the need for differencing.
  - Combining the forecast with actual values for side-by-side comparison.

- **Evaluation Metrics:**
  - **MAE** (Mean Absolute Error)  
  - **MSE** (Mean Squared Error)  
  - **MAPE** (Mean Absolute Percentage Error)

- **Visualization:**
  - Forecast plotted alongside the original sales data.
  - Vertical line marks the **train-test split**.

- **Parameters Used:**
  - `trend='add'` – additive trend component  
  - `seasonal='add'` – additive seasonal component  
  - `seasonal_periods=12` – reflects yearly seasonality in monthly data

- **Models were run for:**
  - `df_men` — *Men Clothing*  
  - `df_women` — *Women Clothing*  
  - `df_other` — *Other Clothing*

- **Outcome:**
  - Generated interpretable and smooth forecasts that adapt to both trend and seasonality in the sales data.

### 🔮 Prophet Forecasting

- A function `run_prophet_forecasting()` was implemented to forecast sales using **Facebook Prophet**, a time series model that automatically handles **trend, seasonality, and holiday effects**.

- **Workflow includes:**
  - Splitting the dataset into training and testing sets (default: 80/20 split).
  - Preparing data for Prophet by renaming columns to `ds` (date) and `y` (target variable).
  - Fitting the Prophet model on the training data.
  - Creating a future dataframe and generating forecasts for the test period.
  - Combining actual and forecasted values for visualization and evaluation.

- **Evaluation Metrics:**
  - **MAE** (Mean Absolute Error)  
  - **MSE** (Mean Squared Error)  
  - **MAPE** (Mean Absolute Percentage Error)

- **Visualization:**
  - Plots the actual vs. predicted sales values over time.
  - Dashed vertical line marks the **train-test split**.

- **Key Features of Prophet:**
  - Automatically detects **changepoints** in trends.
  - Handles **missing data** and **non-linear trends**.
  - Built-in support for **monthly/weekly seasonality** (configured via `freq='M'` in this case).

- **Models were run for:**
  - `df_men` — *Men Clothing Sales*  
  - `df_women` — *Women Clothing Sales*  
  - `df_other` — *Other Clothing Sales*

- **Outcome:**
  - Produced flexible forecasts that capture both seasonality and trend shifts without extensive manual tuning.

### Conclusion - Model Selection Note

After evaluating **SARIMA**, **Exponential Smoothing (Holt-Winters)**, and **Prophet** across three product categories (Men, Women, Other Clothing), the following results were observed:

| Category         | Best Model            | MAPE  | MAE   | MSE    |
|------------------|------------------------|-------|--------|---------|
| **Men Clothing**   | Exponential Smoothing | 3.62% | 25.70 | 778     |
| **Women Clothing** | Exponential Smoothing | 3.78% | 120.97| 28,184  |
| **Other Clothing** | Exponential Smoothing | 5.11% | 62.89 | 6,312   |

- **Exponential Smoothing outperformed SARIMA and Prophet across all three categories** in terms of MAE, MSE, and MAPE.
- It delivered **more accurate and stable forecasts** without requiring differencing or complex hyperparameter tuning like SARIMA.
- Prophet produced reasonable forecasts but underperformed compared to Exponential Smoothing in all metrics.

**Recommendation:**  
Use **Exponential Smoothing (Holt-Winters)** as the primary forecasting model for all three product categories.

**Next Steps:**
- Consider grid-searching `trend`, `seasonal`, and `seasonal_periods` parameters to further optimize the Exponential Smoothing model.
- Explore adding external regressors (holidays, promotions) if revisiting Prophet for complex seasonality.
- Validate results with time series cross-validation to ensure long-term stability.


