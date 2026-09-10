# Time Series Foundations

This portfolio section develops the foundations required before applying deep-learning models to forecasting.

The goal is to show that time-series modeling is not simply:

```text
load data
→ train LSTM
→ report RMSE
```

A valid forecasting workflow requires understanding temporal structure, chronological splitting, leakage, windowing, baselines, and evaluation before model complexity is introduced.

## Learning Progression

<img width="2682" height="614" alt="01_time_series_learning_progression" src="https://github.com/user-attachments/assets/8578aa20-a5f4-4f89-9cf8-6917d6b3e5d3" />


```text
Time Index
    ↓
Temporal Frequency
    ↓
Trend / Seasonality / Noise
    ↓
Lags
    ↓
Autocorrelation
    ↓
Stationarity
    ↓
Missingness / Resampling
    ↓
Chronological Splitting
    ↓
Leakage-Free Scaling
    ↓
Windowing
    ↓
Univariate / Multivariate
    ↓
Single-Step / Multi-Step
    ↓
Forecasting Baselines
    ↓
Evaluation Metrics
    ↓
Walk-Forward Evaluation
    ↓
Prediction Uncertainty
```

## Repository Structure

```text
Time Series Foundations/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01_time_series_structure_and_exploration.ipynb
├── 02_temporal_preprocessing_windowing_and_features.ipynb
├── 03_forecasting_baselines_and_evaluation.ipynb
│
├── data/
│   ├── README.md
│   └── synthetic_hourly_demand.csv
```

## Notebook 1 — Time Series Structure and Exploration From Scratch

This notebook develops the temporal structure before forecasting is introduced.

Progression:

```text
What is a time series?
    ↓
Time index
    ↓
Frequency
    ↓
Regular vs irregular observations
    ↓
Trend
    ↓
Daily and weekly seasonality
    ↓
Lags
    ↓
Manual autocorrelation
    ↓
Rolling statistics
    ↓
Stationarity
    ↓
Differencing
    ↓
Missing values
    ↓
Missing timestamps
    ↓
Interpolation
    ↓
Resampling
    ↓
Exogenous variables
```

Portfolio evidence:

- explains why temporal ordering matters,
- inspects timestamp types and frequency,
- examines the same series at multiple time scales,
- creates lags manually before using `shift()`,
- calculates lag correlation manually,
- distinguishes missing values from missing timestamps,
- explains stationarity and differencing,
- introduces external covariates and forecast-time availability.

## Notebook 2 — Temporal Preprocessing, Windowing, and Features From Scratch

This notebook converts a time series into a valid supervised-learning problem.

Progression:

```text
Temporal leakage
    ↓
Chronological splitting
    ↓
Training-only scaling
    ↓
Lag features
    ↓
Leakage-safe rolling features
    ↓
Calendar features
    ↓
Cyclical encoding
    ↓
Lookback
    ↓
Forecast horizon
    ↓
Stride
    ↓
Manual windowing
    ↓
Single-step targets
    ↓
Multi-step targets
    ↓
Multivariate windows
```

Portfolio evidence:

- demonstrates why random splitting is inappropriate for forecasting,
- fits scaling parameters using training data only,
- explains future-feature leakage,
- builds sliding windows manually,
- constructs model-ready array shapes,
- distinguishes recursive and direct multi-step forecasting,
- discusses split-boundary context and target alignment.

## Notebook 3 — Forecasting Baselines and Evaluation From Scratch

This notebook establishes the benchmark that later deep-learning models must beat.

Progression:

```text
Naive forecast
    ↓
Seasonal naive
    ↓
Moving average
    ↓
Forecast errors
    ↓
MAE
    ↓
MSE / RMSE
    ↓
MAPE / sMAPE
    ↓
MASE
    ↓
Residual analysis
    ↓
Walk-forward evaluation
    ↓
Rolling-origin evaluation
    ↓
Horizon-wise error
    ↓
Prediction intervals
    ↓
Coverage
```

Portfolio evidence:

- implements baseline forecasts explicitly,
- implements major forecasting metrics from scratch,
- analyzes residuals and residual autocorrelation,
- evaluates accuracy by forecast horizon,
- creates a simple empirical uncertainty interval,
- defines a fair benchmark table for the next deep-learning section.

## Dataset

The repository contains a reproducible synthetic hourly demand dataset with:

- hourly timestamps,
- demand,
- temperature,
- humidity,
- trend,
- daily seasonality,
- weekly seasonality,
- random noise,
- several deliberately missing demand measurements.

The synthetic dataset is intentional. It lets the notebooks connect known data-generating structure to trend, seasonality, lag, autocorrelation, leakage, windowing, and evaluation.

Later applied forecasting projects can use real public or research datasets.

## Why This Folder Is Separate From Deep Learning for Time Series Forecasting

This folder answers:

> How should a forecasting problem be structured, prepared, and evaluated?

The next folder answers:

> Which deep-learning architectures improve on that valid forecasting benchmark?

Separating the two sections makes the learning progression visible.


## Visual Explanations

Static diagrams explain:

- overall learning progression,
- time-series components,
- chronological splitting,
- sliding-window transformation,
- forecasting evaluation logic.

The notebooks also generate:

- full-series plots,
- seasonal profiles,
- rolling-statistic plots,
- autocorrelation comparisons,
- split visualizations,
- baseline forecasts,
- residual plots,
- error-by-horizon plots,
- prediction intervals.

## Skills Demonstrated

- temporal data structures,
- frequency and resampling,
- trend and seasonality,
- lag analysis,
- autocorrelation,
- stationarity and differencing,
- missing temporal data,
- leakage prevention,
- chronological splitting,
- training-only normalization,
- temporal feature engineering,
- supervised window construction,
- multivariate forecasting setup,
- multi-step forecasting setup,
- forecasting baselines,
- forecast metrics,
- walk-forward validation,
- uncertainty and interval coverage.


The next section will apply Dense networks, 1D CNNs, RNNs, LSTMs, GRUs, TCNs, Transformers, and Time-Series Foundation Models to the forecasting framework developed here.
