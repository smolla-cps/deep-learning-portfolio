# Natural Language Processing and Text Classification

This portfolio section develops text processing from raw documents to classical and neural text-classification models.

## Repository Structure

```text
10. Natural Language Processing and Text Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_gutenberg_text_preprocessing_bag_of_words_classification.ipynb
├── 02_gutenberg_learned_pretrained_embeddings_text_classification.ipynb
└── 03_glove_pretrained_embeddings_similarity_word_analogies.ipynb
```

## Learning Progression

```text
Raw text
→ cleaning
→ tokenization
→ vocabulary
→ numerical text representation
    ├── bag-of-words / n-grams / TF-IDF
    └── integer sequences
          → padding
          → embeddings
→ text classification
→ pretrained embedding analysis
```

## Notebook 1: Text Representation and Classical Text Classification

This notebook develops the classical text-processing pipeline using Project Gutenberg paragraphs.

It includes:

- raw paragraph collection
- text cleaning
- word tokenization
- vocabulary construction
- manual word-count matrices
- CountVectorizer
- train-fitted vocabulary construction
- rare/common word filtering
- binary word-presence features
- unigram and bigram features
- TF-IDF
- K-nearest neighbors
- logistic regression
- dense neural network
- decision tree
- random forest
- Multinomial, Gaussian, and Bernoulli Naive Bayes
- support vector classification
- confusion matrices
- random-forest word-importance analysis

## Notebook 2: Learned and Pretrained Embeddings for Text Classification

This notebook moves from sparse word-count features to ordered integer sequences and dense embeddings.

It includes:

- Project Gutenberg text
- train/test splitting
- tokenization
- vocabulary inspection
- integer sequences
- sequence padding and truncation
- learned embedding layers
- Dense text classification
- multi-layer Conv1D classification
- compact Conv1D classification
- convolution kernel-size comparison
- LSTM text classification
- learned-embedding cosine similarity
- GloVe download and parsing
- pretrained embedding-matrix construction
- frozen and trainable pretrained embeddings
- pretrained Conv1D classification
- pretrained LSTM classification
- three-book and four-book experiments
- saved training curves, accuracy results, and confusion matrices from the source experiments

## Notebook 3: Pretrained GloVe Embeddings and Word Analogies

This notebook studies the semantic structure of pretrained vectors directly.

It includes:

- GloVe embeddings
- embedding-vector inspection
- cosine similarity
- university similarity
- country similarity
- book-character similarity
- nearest-word search using Euclidean distance
- nearest-word search using cosine similarity
- word analogy arithmetic
- geographic, gender, object, science, cybersecurity, and sports analogy examples

## Why Multiple Text Representations?

A bag-of-words representation records which words occur but loses most word order. N-grams preserve short local word sequences. Integer sequences preserve the full retained order of the tokens and can be passed through an embedding layer.

An embedding layer converts integer word indices into dense vectors. Those vectors can be learned from the classification task or initialized from pretrained vectors such as GloVe.

## Datasets

The text-classification notebooks use paragraphs from four Project Gutenberg books:

- *The Call of the Wild* — Jack London
- *Dracula* — Bram Stoker
- *The Adventures of Sherlock Holmes* — Arthur Conan Doyle
- *Twenty Thousand Leagues under the Sea* — Jules Verne

Some experiments use the first three books, while the extended experiments use all four.

The pretrained embedding notebooks use the Stanford GloVe 6B vectors.

## Portfolio Progression

```text
Text Representation and Classical Classification
→ Learned Word Embeddings
→ Neural Text Classification
→ Pretrained Word Embeddings
→ Sequence Models
→ Attention and Transformer Architectures
```

This section establishes how raw language becomes numerical model input before moving to recurrent and attention-based sequence architectures.

## Running the Notebooks

Google Colab is recommended. Run each notebook from top to bottom in a fresh runtime.

The notebooks download Project Gutenberg text and GloVe vectors from their public sources, so an internet connection is required during the first run.

## Requirements

See `requirements.txt`.
