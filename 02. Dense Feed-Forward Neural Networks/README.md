# Dense Feed-Forward Neural Networks

This portfolio section develops fully connected neural networks after the foundations of gradient descent and backpropagation. The progression moves from MNIST architecture studies to tabular prediction and then to CIFAR-10, where the limitations of dense networks on spatial image data become clear.

## Learning Progression

```text
Gradient Descent & Backpropagation
        ↓
TensorFlow Linear / Logistic Baselines
        ↓
Single and Multilayer Dense Networks
        ↓
One and Two Hidden Layers
        ↓
Activation and Loss Function Choices
        ↓
Hyperparameter Sensitivity
        ↓
Dense Regression
        ↓
CIFAR-10 Dense Classification
        ↓
Feature-Order / Spatial-Structure Limitation
        ↓
Convolutional Neural Networks
```

## Notebooks

### 01. MNIST Dense Network Architectures
`01_mnist_dense_network_architectures.ipynb`

Develops dense-network classification on MNIST from linear and logistic TensorFlow baselines to deeper feed-forward architectures.

**Topics covered**
- Linear and logistic TensorFlow models
- Quadratic feature expansion
- Additional dense layers
- Sigmoid, ReLU, and softmax activations
- Mean-squared error and cross-entropy
- One-hidden-layer networks
- Two-hidden-layer networks
- One- and two-layer linear/sigmoid comparisons
- Model summaries and visual prediction inspection
- Training and test accuracy
- Confusion matrices
- Hyperparameter sensitivity
- Epoch, batch-size, and learning-rate experiments
- Random feature-column shuffling
- Dense-network design and training considerations

### 02. California Housing Dense Regression
`02_california_housing_dense_regression.ipynb`

Develops dense networks for tabular prediction. The notebook begins with grade-based TensorFlow regression/classification and then uses the California Housing dataset for a broader dense-regression architecture study.

**Topics covered**
- Grade-based linear regression
- Input/output scaling
- Binary logistic classification
- Multiclass classification
- Hidden-layer tabular models
- California Housing
- StandardScaler and MinMaxScaler
- Baseline dense regression
- Additional linear layers
- Nonlinear hidden layers
- Increasing network width
- Increasing network depth
- MSE and MAE evaluation

### 03. CIFAR-10 Dense Feed-Forward Neural Network
`03_cifar10_dense_ffnn.ipynb`

Applies a dense feed-forward neural network to CIFAR-10 and examines the consequences of treating flattened image pixels as independent input attributes.

**Topics covered**
- CIFAR-10 preprocessing
- RGB image visualization
- Dense feed-forward architecture
- Training and validation behavior
- Approximately 54% source-reported accuracy
- Overfitting behavior around epochs 20–25
- Random input-column permutation
- Visualization of shuffled images
- Performance comparison after shuffling
- Limitation of dense networks for spatial image structure
- Motivation for convolutional neural networks

## Repository Structure

```text
3. Dense Feed-Forward Neural Networks/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── grades.xlsx
│
├── 01_mnist_dense_network_architectures.ipynb
├── 02_california_housing_dense_regression.ipynb
└── 03_cifar10_dense_ffnn.ipynb
```

## Environment

```bash
pip install -r requirements.txt
```

MNIST and CIFAR-10 are loaded through TensorFlow/Keras. California Housing is loaded through scikit-learn. The grade-based experiments use the included `data/grades.xlsx` file.

## Next Portfolio Section

The CIFAR-10 experiments show a key limitation of fully connected networks: flattening an image does not explicitly preserve or exploit local spatial relationships. The next section therefore moves to **Convolutional Neural Networks (CNNs)**.
