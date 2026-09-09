# Adversarial Machine Learning

This portfolio section studies adversarial examples and model sensitivity to input perturbations using PyTorch and TensorFlow/Keras.

## Repository Structure

```text
9. Adversarial Machine Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mnist_pytorch_adversarial_attacks.ipynb
├── 02_cifar10_pytorch_adversarial_attacks.ipynb
└── 03_tensorflow_keras_adversarial_example_generation.ipynb
```

## Notebook 1: MNIST Adversarial Attacks with PyTorch

Includes the original MNIST feed-forward classifier experiments, random-noise robustness, gradient visualization, FGSM, targeted FGSM, Basic Iterative Method, targeted iterative attacks, alternate thresholds, alternate target classes, and the additional two-hidden-layer network variant.

## Notebook 2: CIFAR-10 Adversarial Attacks with PyTorch

Includes CIFAR-10 preprocessing, the convolutional classifier, class-wise image sampling, random noise, FGSM, targeted FGSM, Basic Iterative Method, targeted iterative attacks, confusion matrices, and saved visual results.

## Notebook 3: TensorFlow/Keras Adversarial Example Generation

Includes selectable MNIST/CIFAR-10 experiments, Gaussian and uniform noise sensitivity, trainable perturbation graphs, clipping, targeted perturbation optimization, sign-based perturbations, and the separate CIFAR-10 VGG-style implementation.

## Datasets

- MNIST
- CIFAR-10

## Frameworks

- PyTorch
- TensorFlow/Keras

## Running the Notebooks

Google Colab with GPU acceleration is recommended. Saved outputs are retained to document the original experimental results. Rerunning cells with random selections or random noise can produce different numerical values and figures.

## Requirements

See `requirements.txt`.
