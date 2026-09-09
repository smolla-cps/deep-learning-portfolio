# Classical ML vs Dense Neural Networks vs CNNs

This portfolio section is centered on **model comparison for image classification**. It evaluates when classical machine-learning methods are sufficient, when dense neural networks are competitive, and when convolutional neural networks provide a clear advantage by preserving spatial image structure.

The experiments compare models, feature representations, optimizers, CNN architectures, regularization methods, augmentation strategies, and output/loss formulations across LFW, MNIST, Fashion-MNIST, CIFAR-10, and CelebA.

## Repository Structure

```text
4. Classical ML vs Dense Neural Networks vs CNNs/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── 01_lfw_classical_vs_dense_vs_cnn.ipynb
├── 02_mnist_fashion_mnist_dense_vs_cnn.ipynb
├── 03_cifar10_classical_vs_dense_vs_cnn.ipynb
└── 04_celeba_cnn_architecture_comparison.ipynb
```

## Comparison Guide

| Notebook | Main Comparison | Datasets |
|---|---|---|
| `01_lfw_classical_vs_dense_vs_cnn.ipynb` | Classical ML vs PCA-based classical models vs dense networks vs CNNs | LFW |
| `02_mnist_fashion_mnist_dense_vs_cnn.ipynb` | Dense networks vs CNNs, SGD vs Adam, sparse vs one-hot labels, compact vs deeper CNNs | MNIST, Fashion-MNIST, CIFAR-10 |
| `03_cifar10_classical_vs_dense_vs_cnn.ipynb` | Classical ML vs dense networks vs CNNs, followed by deeper comparisons of regularization and augmentation | CIFAR-10 |
| `04_celeba_cnn_architecture_comparison.ipynb` | CNN depth, pooling, batch size, normalization, augmentation, API, and multi-label objective comparisons | CelebA |

## What Is Compared

### Classical ML vs Neural Networks
- K-Nearest Neighbors
- Random Forest
- Multinomial Naive Bayes
- Decision Tree
- Logistic Regression
- Support Vector Classifier
- Dense feed-forward neural networks
- Convolutional neural networks

### Feature Representation
- Raw image pixels
- Flattened image vectors
- PCA-reduced features
- Spatial image tensors
- Reshaped PCA feature maps

### Optimization and Architecture
- SGD with momentum vs Adam
- Dense network vs compact CNN
- Compact CNN vs deeper VGG-style CNN
- Four convolutional blocks vs five blocks
- Max pooling vs average pooling
- Different batch sizes

### Generalization Strategies
- Without vs with dropout
- Without vs with batch normalization
- Original vs mirrored training data
- Fixed learning rate vs learning-rate scheduling
- Without vs with image augmentation
- Individual vs combined augmentation layers
- Original vs mirrored vs ensemble test-time prediction

### Output and Loss Formulations
- Sparse categorical labels vs one-hot labels
- Softmax vs sigmoid outputs
- Categorical cross-entropy vs binary cross-entropy
- Single-label vs multi-label classification

## Selected Comparative Results

Recorded runs from the included experiments illustrate the differences between approaches:

- **LFW:** PCA + SVC reached about **0.7974 accuracy**, while a PCA-based dense network reached about **0.8231**.
- **MNIST:** the CNN reached about **0.9933 validation accuracy**, compared with about **0.9849** for the dense Adam model.
- **Fashion-MNIST:** CNN validation accuracy reached about **0.9199**, compared with about **0.8999** for the dense Adam model.
- **CIFAR-10:** the dense baseline remained near **0.51 peak validation accuracy**, while the compact CNN reached about **0.71**.
- **CIFAR-10 augmentation/test-time ensemble:** a recorded run reached about **0.9006 accuracy**.

These comparisons show that dense models remain effective for simpler image datasets, while CNNs become increasingly advantageous as spatial structure and image complexity increase.

## Data

MNIST, Fashion-MNIST, CIFAR-10, and LFW are loaded directly through TensorFlow or scikit-learn.

CelebA is not included in the repository. See [`data/README.md`](data/README.md) for the expected local file structure.

## Running the Notebooks

Install the dependencies:

```bash
pip install -r requirements.txt
```

A GPU runtime is recommended for the deeper CIFAR-10 and CelebA CNN comparisons.
