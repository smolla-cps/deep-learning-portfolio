# Sequence Modeling

This portfolio section develops sequence models from recurrent architecture fundamentals to image-sequence classification, multivariate time-series forecasting, and LSTM-based text classification.

## Repository Structure

```text
11. Sequence Modeling/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mnist_rnn_lstm_gru_sequence_classification.ipynb
├── 02_jena_climate_rnn_lstm_gru_forecasting.ipynb
└── 03_gutenberg_lstm_learned_pretrained_embeddings.ipynb
```

## Learning Progression

```text
Why sequence order matters
→ recurrent hidden state
→ Simple RNN
→ LSTM
→ GRU
→ sequence classification
→ time-series forecasting
→ text sequence classification
→ learned and pretrained embeddings with LSTM
```

## Notebook 1: MNIST RNN, LSTM, and GRU Sequence Classification

Treats each 28 × 28 MNIST image as a sequence of 28 rows and compares:

- Simple RNN
- LSTM
- GRU
- parameter counts
- training/validation behavior
- test performance
- confusion matrix
- example predictions

The notebook also demonstrates why sequential order and recurrent state provide information that a fixed order-independent representation can lose.

## Notebook 2: Jena Climate RNN, LSTM, and GRU Forecasting

Applies recurrent models to multivariate temporal data:

- hourly climate measurements
- chronological train/validation/test splitting
- normalization from training statistics
- 120-hour input sequences
- 24-hour-ahead temperature forecasting
- Simple RNN
- LSTM
- GRU
- MAE/MSE comparison
- predicted vs. actual temperature

## Notebook 3: Gutenberg LSTM with Learned and Pretrained Embeddings

Applies LSTM sequence modeling to paragraph classification from four Project Gutenberg books:

- raw paragraph collection
- training-only vocabulary construction
- tokenization
- integer sequences
- padding/truncation
- learned embedding baseline
- GloVe 50d, 100d, and 200d experiments
- trainable pretrained embeddings
- frozen pretrained embeddings
- two-stage frozen-to-trainable fine-tuning
- LSTM paragraph classification
- confusion matrices and test accuracy

The final implementation uses separate training, validation, and test data so early stopping does not use the test set.

## What RNN, LSTM, and GRU Learn

A recurrent model processes an ordered sequence one step at a time. The hidden state carries information from earlier time steps.

- **RNN** learns how the current input and previous hidden state should be combined.
- **LSTM** adds learned gates that control what information should be remembered, updated, or forgotten.
- **GRU** provides a simpler gated recurrent mechanism with update and reset gates.

The same recurrent ideas are demonstrated on image rows, climate observations, and word sequences.

## Datasets

- **MNIST** for image-sequence classification
- **Jena Climate** for multivariate time-series forecasting
- **Project Gutenberg books** for paragraph sequence classification
- **GloVe 6B** for pretrained word embeddings

## Portfolio Progression

```text
10. Natural Language Processing and Text Classification
→ 11. Sequence Modeling
→ 12. Attention and Transformer Architectures
```

This section establishes recurrent sequence processing before moving to attention-based sequence architectures.

## Running the Notebooks

Google Colab with GPU acceleration is recommended.

Run each notebook from top to bottom in a fresh runtime. The Gutenberg notebook downloads GloVe vectors during execution.

## Requirements

See `requirements.txt`.
