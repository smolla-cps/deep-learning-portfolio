# Autoencoders

This portfolio section develops autoencoders from decoder fundamentals to practical image reconstruction, denoising, and latent-space selection.

## Repository Structure

```text
7. Autoencoders/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_transposed_convolution_and_upsampling.ipynb
├── 02_mnist_dense_convolutional_denoising_autoencoders.ipynb
└── 03_cifar10_convolutional_autoencoder_latent_selection.ipynb
```

## Notebook 1: Transposed Convolution and Upsampling

Builds decoder operations from first principles with NumPy:

- valid 2D convolution
- transposed convolution
- overlapping contribution accumulation
- multiple input/kernel examples
- explicit row and column upsampling

This notebook provides the decoder intuition used by the later convolutional autoencoders.

## Notebook 2: MNIST Autoencoders

Develops several autoencoder applications on MNIST:

- dense autoencoder
- 32-dimensional latent representation
- progressive reconstruction visualization
- train/test reconstruction MSE
- convolutional encoder
- transposed-convolution decoder
- sigmoid + binary-crossentropy reconstruction
- clipped-ReLU + MSE reconstruction
- reconstruction visualization
- downstream CNN classification on original images
- downstream classification on reconstructed images
- noisy-image generation
- convolutional-autoencoder denoising
- noisy vs. denoised classification comparison

## Notebook 3: CIFAR-10 Convolutional Autoencoder

Studies latent-space dimensionality on color images:

- CIFAR-10 normalization
- separate train, validation, and test sets
- convolutional encoder
- transposed-convolution decoder
- latent dimensions 8, 16, 32, and 64
- L1 activity regularization
- explicit latent L2 penalty
- early stopping
- test reconstruction MSE
- compactness-aware ranking criterion
- selected-model reconstruction visualization
- MSE vs. latent-dimension comparison

## Technical Corrections Incorporated

The MNIST MSE-based convolutional-autoencoder experiment now evaluates the convolutional encoder/decoder that was actually trained rather than reusing the earlier dense autoencoder variables. The reconstructed-classification and denoising sections therefore use the correct current convolutional reconstruction.

For CIFAR-10, model selection and early stopping use a validation subset created from the original training data. The official test set is reserved for final reconstruction MSE so the test set does not influence training decisions.

## Datasets

The notebooks use datasets available directly through TensorFlow/Keras:

- **MNIST** for dense, convolutional, and denoising autoencoders
- **CIFAR-10** for color-image reconstruction and latent-space selection

No dataset files need to be stored in the repository.

## Portfolio Progression

```text
Dense / CNN Foundations
→ Non-Sequential Architectures
→ Transfer Learning
→ Autoencoders
→ Image Segmentation
```

The decoder concepts and latent representations developed here provide a natural foundation for encoder-decoder segmentation architectures.

## Running the Notebooks

Google Colab with GPU acceleration is recommended for the TensorFlow notebooks. Run each notebook from top to bottom in a fresh runtime.

## Requirements

See `requirements.txt` for the main dependencies.
