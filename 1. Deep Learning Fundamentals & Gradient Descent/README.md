# Deep Learning Fundamentals & Gradient Descent

This portfolio section bridges classical machine learning and deep learning. It starts with familiar image-classification baselines, moves through linear and logistic models, and then develops the optimization concepts used to train neural networks.

The progression is designed to show how traditional machine-learning ideas lead naturally to gradient-based learning, automatic differentiation, and backpropagation.

## Learning Progression

```text
Classical Machine Learning
        ↓
Image Classification Baselines
        ↓
Linear and Logistic Models
        ↓
TensorFlow / Keras
        ↓
Gradient Descent
        ↓
Automatic Differentiation
        ↓
Backpropagation
        ↓
Neural Network Fundamentals
```

## Notebooks

### 01. Image Classification Baselines
`01_image_classification_baselines.ipynb`

Establishes non-neural image-classification baselines before learned neural representations are introduced.

**Topics covered**
- Constant-class baseline
- ZeroR
- One-nearest-neighbor
- Vectorized Euclidean-distance computation
- k-nearest neighbors
- MNIST
- Fashion-MNIST
- CIFAR-10
- Attribute-order experiments
- Principal component analysis

This notebook provides the connection between the machine-learning portfolio and the beginning of the deep-learning progression.

---

### 02. Linear and Logistic Models
`02_linear_logistic_models.ipynb`

Develops the transition from manually implemented linear and logistic classifiers to TensorFlow/Keras models.

**Topics covered**
- Binary MNIST classification
- Linear regression as a classifier
- Classification thresholds
- Logistic regression
- Sigmoid activation
- Manual gradient-based training
- scikit-learn comparison
- TensorFlow/Keras Dense models
- Quadratic feature expansion
- Softmax activation
- Mean-squared error
- Cross-entropy loss
- Stochastic gradient descent
- Momentum
- Adam
- Training and evaluation

This notebook connects traditional classification models with the optimization tools and framework abstractions used in deep learning.

---

### 03. Gradient Descent and Backpropagation
`03_gradient_descent_and_backpropagation.ipynb`

Focuses on the optimization principles that make neural-network learning possible.

**Topics covered**
- Multivariable linear models
- Mean-squared-error loss
- Explicit gradient computation
- Weight and bias updates
- Gradient descent with NumPy
- PyTorch tensors
- Automatic differentiation
- Computational graphs
- Sigmoid activation
- Binary classification
- Binary cross-entropy
- Chain rule
- Backpropagation

The notebook progresses from manually calculated gradients to PyTorch autograd, showing how deep-learning frameworks automate the same mathematical learning process.

## Repository Structure

```text
1. Deep Learning Fundamentals & Gradient Descent/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── grades.xlsx
│
├── 01_image_classification_baselines.ipynb
├── 02_linear_logistic_models.ipynb
└── 03_gradient_descent_and_backpropagation.ipynb
```

## Portfolio Progression

This section establishes the foundation for the later deep-learning topics.

```text
Deep Learning Fundamentals & Gradient Descent
        ↓
Dense Feed-Forward Neural Networks
        ↓
Convolutional Neural Networks
        ↓
Dense vs. Convolutional Networks
        ↓
Segmentation
        ↓
Advanced / Non-Sequential Architectures
        ↓
Transfer Learning
        ↓
Autoencoders
        ↓
Sequence Models
        ↓
Natural Language Processing
        ↓
Attention and Transformer Architectures
        ↓
Adversarial Machine Learning
```

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The gradient-descent notebook uses the included `data/grades.xlsx` file. Standard benchmark datasets such as MNIST, Fashion-MNIST, and CIFAR-10 are loaded through their corresponding Python libraries.
