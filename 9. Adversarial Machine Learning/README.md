# Adversarial Machine Learning

This portfolio section studies how image classifiers respond to intentionally designed input perturbations. It progresses from random-noise baselines to gradient-based attacks in PyTorch and trainable perturbation optimization in TensorFlow/Keras.

## Repository Structure

```text
9. Adversarial Machine Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mnist_pytorch_adversarial_attacks.ipynb
├── 02_cifar10_pytorch_adversarial_attacks.ipynb
└── 03_tensorflow_keras_adversarial_perturbation_optimization.ipynb
```

## Notebook 1: MNIST Adversarial Attacks with PyTorch

Develops adversarial attacks on a feed-forward MNIST classifier:

- one-hidden-layer classifier
- two-hidden-layer classifier variant
- clean test accuracy
- uniform random-noise robustness
- input-gradient calculation
- gradient and gradient-sign visualization
- untargeted FGSM
- FGSM accuracy-threshold variation
- targeted FGSM
- target-class variation
- untargeted Basic Iterative Method
- targeted Basic Iterative Method
- additional targeted-BIM class experiment
- confusion matrices
- perturbation magnitude analysis
- original/adversarial/noise visualization

## Notebook 2: CIFAR-10 Adversarial Attacks with PyTorch

Extends the same attack progression to color images and a convolutional classifier:

- CIFAR-10 channels-first preprocessing for PyTorch
- convolutional neural-network training
- class-wise image sampling
- random perturbation visualization
- random-noise attack
- untargeted FGSM
- targeted FGSM
- untargeted BIM
- targeted BIM
- attack confusion matrices
- three-channel perturbation visualization

## Notebook 3: TensorFlow/Keras Adversarial Perturbation Optimization

Presents a distinct adversarial-generation approach in which perturbation values are trainable model parameters:

- selectable MNIST or CIFAR-10 workflow
- compact CNN and VGG-style CNN baselines
- Gaussian-noise sensitivity
- uniform-noise sensitivity
- trainable perturbation layer
- unclipped and clipped perturbation graphs
- frozen target classifier
- targeted perturbation optimization
- L2 perturbation regularization
- second-most-likely and least-likely target strategies
- original/adversarial/perturbation visualization
- sign-based perturbation direction

## Attack Progression

```text
Random Noise
→ Input Gradients
→ FGSM
→ Targeted FGSM
→ Basic Iterative Method
→ Targeted BIM
→ Trainable Perturbation Optimization
```

Random noise provides a non-adaptive baseline. FGSM uses a single loss-gradient direction, while BIM repeatedly updates the adversarial input. Targeted variants optimize toward a chosen class. The TensorFlow/Keras workflow provides another perspective by directly learning a perturbation while keeping the classifier fixed.

## Datasets

The notebooks use:

- **MNIST** for feed-forward-network attack experiments
- **CIFAR-10** for convolutional-network attack experiments

Both datasets are downloaded through TensorFlow/Keras.

## Frameworks

This section intentionally includes both:

- **PyTorch** for gradient-based adversarial attacks
- **TensorFlow/Keras** for trainable perturbation optimization

Using both frameworks demonstrates the same adversarial-learning concepts through different computational-graph and optimization interfaces.

## Technical Notes

The PyTorch classifiers use integer class labels with `CrossEntropyLoss`, which is the standard formulation for multiclass classification. Gradient buffers are cleared between iterative attack steps so each BIM update uses the current input gradient rather than accumulated gradients.

The CIFAR-10 PyTorch classifier computes the dense-layer input size from the known 32×32 image progression rather than relying on a hard-coded value that does not match the final feature-map shape.

The TensorFlow/Keras classifier uses a validation split from the training data during model fitting. The official test set is therefore reserved for final evaluation and adversarial experiments rather than being used as the validation set during training.

Modern TensorFlow/Keras imports and named perturbation-layer access are used so the notebooks remain compatible with current runtimes.

## Portfolio Progression

```text
CNN Fundamentals
→ Advanced / Non-Sequential Architectures
→ Transfer Learning
→ Autoencoders
→ Image Segmentation
→ Adversarial Machine Learning
```

## Running the Notebooks

Google Colab with GPU acceleration is recommended, particularly for CIFAR-10 model training. Run each notebook from top to bottom in a fresh runtime.

## Requirements

See `requirements.txt` for the main dependencies.
