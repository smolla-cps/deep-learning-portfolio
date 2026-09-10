# 15. Deep Learning for Time Series Forecasting

This portfolio section continues directly from ** Time Series Foundations** directory.

The previous folder established how a forecasting problem must be structured:

```text
time order
→ chronological splitting
→ leakage-free scaling
→ windowing
→ baselines
→ evaluation
```

This folder now studies how increasingly advanced deep-learning architectures solve that valid forecasting problem.

The emphasis is not simply on obtaining a low RMSE. The notebooks compare how different architectures represent temporal information, how forecasting difficulty changes across horizons, and how pretrained time-series foundation models change the traditional task-specific training workflow.

## Learning Progression

<img width="2862" height="650" alt="01_deep_time_series_progression" src="https://github.com/user-attachments/assets/dc0bb909-f041-4083-a6ef-534ad14b6943" />


```text
Seasonal Baseline
        ↓
Dense Network
        ↓
1D CNN
        ↓
RNN
        ↓
LSTM
        ↓
GRU
        ↓
Temporal Convolutional Network
        ↓
Multivariate Forecasting
        ↓
Multi-Step Forecasting
        ↓
Transformer Forecasting
        ↓
Patch-Based Transformer
        ↓
Time-Series Foundation Model
        ↓
Zero-Shot Forecasting
```

## Repository Structure

```text
16. Deep Learning for Time Series Forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01_dense_cnn_rnn_lstm_gru_forecasting.ipynb
├── 02_temporal_convolutional_networks.ipynb
├── 03_multivariate_and_multistep_forecasting.ipynb
├── 04_transformers_and_time_series_foundation_models.ipynb
│
├── data/
│   ├── README.md
│   └── synthetic_hourly_demand.csv

```

# Notebook 1 — Dense, CNN, RNN, LSTM, and GRU Forecasting

This notebook begins with a controlled architecture comparison.

All models use the same:

```text
dataset
chronological split
training-only scaling
24-hour lookback
one-step target
training procedure
test period
MAE / RMSE
```

Progression:

```text
Seasonal Naive
      ↓
Dense Neural Network
      ↓
1D CNN
      ↓
Simple RNN
      ↓
LSTM
      ↓
GRU
      ↓
Fair Architecture Comparison
```

The notebook demonstrates:

- why a Dense network is a useful neural baseline,
- how a 1D CNN learns local temporal patterns,
- how recurrent hidden state differs from convolution,
- why LSTM and GRU are evaluated after a simple RNN,
- tensor shape differences across architectures,
- parameter counts,
- training-time comparison,
- validation learning curves,
- inverse scaling,
- forecast visualization,
- residual interpretation.

The central portfolio question is:

> Does added architectural complexity produce meaningful forecasting improvement over a strong temporal baseline?

# Notebook 2 — Temporal Convolutional Networks

This notebook develops TCN mechanics from scratch.

Progression:

```text
Ordinary 1D CNN
      ↓
Forecast Causality
      ↓
Left-Only Padding
      ↓
Causal Convolution
      ↓
Dilation
      ↓
Receptive Field
      ↓
Residual Connections
      ↓
TCN
      ↓
Forecast Evaluation
```

The notebook demonstrates:

- why symmetric convolution can leak future information,
- explicit causal padding,
- an experimental causality check,
- dilated convolution,
- receptive-field calculation,
- residual TCN blocks,
- long-context forecasting,
- comparison with seasonal naive,
- receptive-field design as a model-selection problem.

# Notebook 3 — Multivariate and Multi-Step Forecasting

This notebook changes the forecasting formulation itself.

The model now predicts:

```text
previous 72 hours
        ↓
next 24 hours
```

and compares univariate and multivariate historical inputs.

Progression:

```text
Single-Step
      ↓
Multi-Step
      ↓
Univariate Input
      ↓
Multivariate Input
      ↓
Direct Multi-Output LSTM
      ↓
Horizon-Wise Evaluation
      ↓
Recursive One-Step Forecasting
      ↓
Error Accumulation
      ↓
Direct vs Recursive
      ↓
Seq2Seq Concept
```

The notebook demonstrates:

- 72-to-24 forecasting windows,
- multiple historical covariates,
- training-only feature scaling,
- cyclical calendar variables,
- direct 24-output LSTM forecasting,
- seasonal-naive multi-step baseline,
- horizon-specific MAE,
- recursive forecasting,
- recursive error accumulation,
- future-covariate leakage,
- sequence-to-sequence forecasting concepts.

# Notebook 4 — Transformers and Time-Series Foundation Models

This is the advanced endpoint of the folder.

Progression:

```text
Historical Window
      ↓
Input Projection
      ↓
Positional Information
      ↓
Self-Attention
      ↓
Transformer Encoder
      ↓
24-Step Forecast
      ↓
Patch-Based Modeling
      ↓
Time-Series Foundation Models
      ↓
Zero-Shot Forecasting
```

The notebook demonstrates:

- how attention is adapted to forecasting,
- why a historical-context encoder does not automatically require a causal mask,
- learned positional embeddings,
- a PyTorch Transformer forecaster,
- horizon-wise Transformer evaluation,
- patch-based time-series representation,
- a small Patch Transformer,
- the conceptual change from task-specific training to large-scale temporal pretraining,
- zero-shot forecasting,
- context length and prediction horizon,
- point forecasts vs quantile forecasts,
- an optional TimesFM zero-shot example,
- fair evaluation of foundation models against simple and trained baselines.

## Why RNN, LSTM, and GRU Are Not Re-Taught From the Beginning

The earlier **Sequence Modeling** portfolio section explains recurrent-network mechanics.

This forecasting section therefore does not repeat all gate equations and sequence-model foundations.

Instead, it asks:

> How do those architectures perform when the task is a rigorously defined temporal forecasting problem?

This avoids repetition while preserving learning progression.

## Why Transformers Are Included Again

The earlier **Attention and Transformer Architectures** section explains:

```text
query
key
value
self-attention
multi-head attention
Transformer blocks
```

This folder does not repeat that theory.

It focuses on:

```text
historical windows
→ temporal embeddings
→ forecasting head
→ multi-horizon output
→ patching
```

which are forecasting-specific applications.

## Time-Series Foundation Models

The final notebook introduces pretrained temporal models as a separate modeling paradigm.

Traditional workflow:

```text
one dataset
→ train model
→ validate
→ test
```

Foundation-model workflow:

```text
large-scale pretraining
→ new unseen time series
→ provide context
→ zero-shot forecast
```

The notebook uses TimesFM as the practical optional example and emphasizes that foundation models must still be evaluated using the same chronological discipline and temporal baselines developed earlier.

Because foundation-model APIs evolve quickly, the optional external section is isolated from the from-scratch educational core.

## Dataset

The repository includes a reproducible hourly demand dataset with:

- demand,
- temperature,
- humidity,
- daily seasonality,
- weekly seasonality,
- trend,
- noise.

The dataset is intentionally manageable so every model can be trained in a notebook while preserving a controlled experimental setup.

A later applied project can repeat the same methodology on a real forecasting dataset.


## Visual Explanations

Static diagrams explain:

- the complete deep temporal progression,
- how different neural architectures view the same window,
- TCN architecture,
- multi-step forecasting,
- Transformer forecasting,
- time-series foundation models.

The notebooks also create:

- validation learning curves,
- actual-vs-predicted plots,
- residual comparisons,
- horizon-wise error plots,
- recursive error-accumulation plots,
- Transformer forecast comparisons,
- optional zero-shot foundation-model forecasts.

## Evaluation Philosophy

Every advanced model should be compared against simple forecasting references.

The portfolio therefore carries forward the baseline mindset from Folder 15:

```text
Seasonal Naive
        ↓
Task-Specific Neural Models
        ↓
Modern Temporal Architectures
        ↓
Foundation Model
```

The goal is not to prove that the newest model always wins.

The goal is to demonstrate:

- valid experimental design,
- architectural understanding,
- honest comparison,
- horizon-aware evaluation,
- computational trade-offs,
- appropriate interpretation.

## Recommended Runtime

Google Colab is recommended.

A GPU is useful for:

- recurrent model comparison,
- TCN training,
- Transformer training,
- optional foundation-model inference.

The educational models are intentionally much smaller than production forecasting systems.

## Core Skills Demonstrated

- deep time-series forecasting,
- neural forecasting baselines,
- 1D temporal convolution,
- recurrent forecasting,
- LSTM and GRU forecasting,
- causal convolution,
- dilation and receptive fields,
- TCN residual blocks,
- multivariate temporal modeling,
- multi-step forecasting,
- recursive forecasting,
- direct multi-output forecasting,
- horizon-wise evaluation,
- Transformer forecasting,
- temporal patching,
- time-series foundation models,
- zero-shot forecasting,
- quantile-forecast interpretation.

## Position in the Deep Learning Portfolio

```text
13. Large Language Models
        ↓
14. Generative Deep Learning
        ↓
15. Time Series Foundations
        ↓
16. Deep Learning for Time Series Forecasting
```

This completes the temporal learning progression from basic forecasting formulation to modern pretrained temporal models.
