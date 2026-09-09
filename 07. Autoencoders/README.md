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

The experiments evaluate autoencoder quality both at the pixel level and through the effect of reconstruction and denoising on downstream classification.

## Notebook 3: CIFAR-10 Convolutional Autoencoder

Studies the relationship between latent-space dimensionality, compression, and reconstruction quality on color images:

- CIFAR-10 normalization
- separate training, validation, and test sets
- convolutional encoder
- transposed-convolution decoder
- latent dimensions 8, 16, 32, and 64
- configurable latent regularization
- explicit latent L2 penalty
- early stopping
- test reconstruction MSE
- compactness-aware ranking criterion
- selected-model reconstruction visualization
- reconstruction comparison across all latent dimensions
- MSE vs. latent-dimension comparison

### Latent-Dimension Results

The final experiment produced the following test reconstruction errors:

| Latent Dimension | Test MSE |
| ---: | ---: |
| 8 | 0.021807 |
| 16 | 0.016571 |
| 32 | 0.012126 |
| 64 | 0.008277 |

Reconstruction error decreases consistently as the latent dimension increases. Smaller latent spaces provide stronger compression but lose more visual detail, while larger latent spaces retain more information and generate sharper reconstructions.

A compactness-aware objective is also evaluated:

$$
J = \mathrm{MSE} + \lambda \left(\frac{\text{latent dimension}}{64}\right)
$$

with `lambda_dim = 1e-3`.

The resulting ranking is:

| Latent Dimension | MSE | J |
| ---: | ---: | ---: |
| 64 | 0.008277 | 0.009277 |
| 32 | 0.012126 | 0.012626 |
| 16 | 0.016571 | 0.016821 |
| 8 | 0.021807 | 0.021932 |

With the current compactness penalty, the reconstruction improvement from the larger latent space outweighs the dimensionality penalty, so the **64-dimensional latent representation is selected**.

## Technical Notes

The MNIST MSE-based convolutional-autoencoder experiment evaluates the convolutional encoder and decoder that are actually trained in that section. The reconstructed-image classification and denoising experiments therefore use the corresponding convolutional reconstructions.

For CIFAR-10, early stopping and model-selection decisions use a validation subset created from the original training data. The official CIFAR-10 test set is reserved for final reconstruction evaluation.

The final latent-dimension sweep uses light latent regularization so larger latent spaces can make use of their additional representational capacity while still retaining a compactness penalty.

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
