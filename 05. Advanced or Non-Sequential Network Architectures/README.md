# Advanced or Non-Sequential Network Architectures

This portfolio section explores neural-network designs that go beyond a simple sequential stack of layers. It develops graph-based model construction with the Keras Functional API, applies those ideas to advanced CIFAR-10 architectures, and then demonstrates flexible training control with Keras callbacks.

## Repository Structure

```text
5. Advanced or Non-Sequential Network Architectures/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mnist_functional_api_and_nonsequential_architectures.ipynb
├── 02_cifar10_advanced_functional_architectures.ipynb
└── 03_cifar10_flexible_training_callbacks.ipynb
```

## Notebook 1: MNIST Functional API and Non-Sequential Architectures

Introduces and develops the architectural concepts behind non-sequential neural networks:

- limitations of strictly sequential models
- Sequential vs Functional API construction
- directed acyclic computation graphs
- parallel convolutional branches
- Inception-style feature extraction
- reusable branch modules
- skip and residual connections
- addition and concatenation
- alternate tensor paths
- three-branch feature fusion
- multiple-output Functional API graphs
- repeated residual combinations
- unused layers
- incompatible tensor shapes
- disconnected graphs

MNIST is used so the network-graph concepts can be studied with relatively low computational cost.

## Notebook 2: CIFAR-10 Advanced Functional Architectures

Applies the same graph-based ideas to a more challenging image-classification dataset:

- functional CNN baseline
- batch normalization and dropout
- parallel-branch CNN
- skip-connection CNN
- combined branch-and-skip architecture
- Inception-style network
- residual network
- residual network with parallel branches
- alternate functional CIFAR-10 baseline

The notebook shows a progression from conventional convolutional models to increasingly flexible computation graphs.

## Notebook 3: CIFAR-10 Flexible Training with Callbacks

Demonstrates training-control techniques used with deeper and more expensive neural networks:

- reusable MNIST / Fashion-MNIST / CIFAR-10 loader
- fixed-epoch baseline training
- mirrored-image augmentation
- separate validation-set construction
- early stopping
- model checkpointing
- best-model reload
- original and mirrored-test prediction combination
- custom performance-threshold callbacks
- multiple stopping thresholds
- multiple patience settings
- `ReduceLROnPlateau`
- minimum-learning-rate variants
- dropout variants
- `LearningRateScheduler`
- combined callback strategies
- `.keras` checkpoint workflow

## Datasets

The notebooks use datasets available directly through TensorFlow/Keras:

- **MNIST** for graph-construction fundamentals
- **CIFAR-10** for advanced convolutional architectures and callback experiments
- **Fashion-MNIST** is also supported by the reusable dataset-loading utility

No dataset files need to be stored in the repository.

## Portfolio Progression

```text
Dense Feed-Forward Networks
→ Convolutional Neural Networks
→ Classical ML vs Dense vs CNN Comparison
→ Advanced / Non-Sequential Network Architectures
→ Transfer Learning
→ Autoencoders
→ Image Segmentation
```

The Functional API concepts developed here provide a foundation for later residual, transfer-learning, encoder-decoder, and U-Net-style architectures.

## Running the Notebooks

Google Colab with GPU acceleration is recommended for the CIFAR-10 experiments. Run each notebook from top to bottom in a fresh runtime.

## Requirements

See `requirements.txt` for the main Python dependencies.
