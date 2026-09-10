# Attention and Transformer Architectures

This portfolio section develops the progression from recurrent sequence-model limitations to attention, self-attention, multi-head attention, positional encoding, and complete Transformer blocks.

## Repository Structure

```text
12. Attention and Transformer Architectures/
├── README.md
├── requirements.txt
├── .gitignore
├── images/
│   ├── 01_recurrent_bottleneck_vs_attention.png
│   ├── 02_query_key_value.png
│   ├── 03_self_attention.png
│   ├── 04_multi_head_attention.png
│   └── 05_transformer_encoder_decoder.png
├── 01_attention_fundamentals_query_key_value.ipynb
├── 02_self_attention_multihead_positional_encoding.ipynb
└── 03_transformer_encoder_decoder_text_classification.ipynb
```

## Learning Progression

```text
limitations of recurrent models
→ attention
→ Query / Key / Value
→ scaled dot-product attention
→ self-attention
→ padding and causal masks
→ multi-head attention
→ positional encoding
→ Transformer encoder
→ Transformer decoder
→ Transformer text classification
```

## Notebook 1: Attention Fundamentals

This notebook explains why attention was introduced and builds the mechanism step by step:

- recurrent encoder-decoder bottleneck
- attention scores
- softmax attention weights
- weighted context vector
- Query, Key, and Value
- scaled dot-product attention
- Keras Attention layer
- attention visualization
- explanation of what the learned attention parameters represent

## Notebook 2: Self-Attention, Multi-Head Attention, and Position

This notebook extends attention to relationships inside one sequence:

- Query, Key, and Value projections
- scaled dot-product self-attention from scratch
- self-attention matrices
- padding masks
- causal masks
- multi-head attention
- attention-head visualization
- sinusoidal positional encoding
- explanation of why position information is required
- recurrent processing vs. self-attention
- quadratic sequence-length cost of standard attention

## Notebook 3: Transformer Architecture and Text Classification

This notebook builds the complete Transformer architecture and applies an encoder to IMDB sentiment classification:

- token and learned-position embeddings
- residual connections
- layer normalization
- position-wise feed-forward networks
- Transformer encoder block
- Transformer decoder block
- masked self-attention
- cross-attention
- IMDB sentiment classification
- Dense baseline
- LSTM baseline
- Transformer encoder classifier
- parameter, accuracy, and training-time comparison
- learned attention-score visualization

## Notebook 4: Gutenberg Transformer Experiments

This notebook adds a second practical Transformer classification problem using four Project Gutenberg books.

It includes:

- raw paragraph collection
- training-only vocabulary construction
- integer sequence representation
- baseline Transformer encoder
- hyperparameter variation with larger embeddings, more attention heads, wider feed-forward layers, and higher dropout
- 100-dimensional pretrained GloVe embeddings
- frozen pretrained token embeddings
- separate training, validation, and test evaluation
- confusion matrices and accuracy comparison
- the recorded result summary from the supplied Transformer solution

The notebook also corrects the undefined `dropout_rate_new` variable from the supplied solution and reserves the final test set for evaluation rather than using it as validation data during training.

## Why Attention Improves Sequence Modeling

Recurrent models pass information through a hidden state. LSTM and GRU improve memory with gates, but sequence processing remains recurrent.

Attention introduces a direct path from the current query to relevant sequence positions.

Self-attention extends this idea so that each token can directly interact with other tokens in the same sequence.

Transformers combine self-attention with position information, residual connections, normalization, and feed-forward networks to model sequences without recurrent hidden-state updates.

## Main Architectural Comparison

| Architecture | Main Strength | Main Limitation |
|---|---|---|
| Dense | Simple fixed-feature learning | No recurrent memory; order may be lost depending on representation |
| CNN / Conv1D | Learns local patterns efficiently | Long-range relationships depend on receptive field |
| RNN | Explicit recurrent state | Long dependencies and sequential computation |
| LSTM / GRU | Gated memory improves long dependencies | Still recurrent and sequential |
| Self-Attention | Direct token-to-token relationships | Standard attention cost grows quadratically with sequence length |
| Transformer | Parallel self-attention with position information | Attention memory/compute cost can be high for long sequences |

## Portfolio Progression

```text
10. Natural Language Processing and Text Classification
→ 11. Sequence Modeling
→ 12. Attention and Transformer Architectures
```

This section completes the transition from recurrent sequence modeling to attention-based sequence modeling.

## Running the Notebooks

Google Colab with GPU acceleration is recommended.

The first two notebooks are lightweight conceptual and coding demonstrations. The third notebook downloads the IMDB dataset and trains Dense, LSTM, and Transformer models.

## Requirements

See `requirements.txt`.
