# Convolutional Neural Networks (CNNs)

This portfolio section develops convolutional neural networks from foundational image
classification to architecture design, training optimization, augmentation, scalable TensorFlow
data pipelines, and pretrained convolutional models.

## Learning Progression

```text
Dense Image Baselines
        ↓
Convolution and Pooling
        ↓
MNIST / Fashion-MNIST / CIFAR-10 CNNs
        ↓
Architecture Design
        ↓
Kernel Size / Stride / Depth
        ↓
Regularization and Early Stopping
        ↓
Dropout / Batch Normalization
        ↓
Learning-Rate Control
        ↓
Image Augmentation
        ↓
tf.data Pipelines
        ↓
Advanced CNN Blocks
        ↓
Pretrained CNN Models
```

## Notebooks

### 01. CNN Fundamentals: MNIST, Fashion-MNIST, and CIFAR-10
`01_mnist_fashion_mnist_cifar10_cnn_fundamentals.ipynb`

**Coverage**
- Dense and CNN baselines
- MNIST
- Fashion-MNIST
- CIFAR-10
- Convolution and max pooling
- Multiple convolution stages
- Dense versus convolutional classification
- Manual forward pass from extracted weights
- CIFAR-10 feature-order sensitivity
- Spatial-structure motivation for CNNs

### 02. CIFAR-10 CNN Architecture Design
`02_cifar10_cnn_architecture_design.ipynb`

**Coverage**
- Baseline CNN
- Reduced network size
- Larger kernels
- Back-to-back convolutions
- Mirrored training and evaluation
- Strided convolution
- Increased convolutional depth
- Increased dense depth
- Combined architecture variants
- Filter-count and optimizer variants

### 03. CIFAR-10 CNN Regularization and Training Optimization
`03_cifar10_cnn_regularization_optimization.ipynb`

**Coverage**
- Early stopping
- Best-weight restoration
- Mirrored training data
- Mirrored test-time prediction
- L1/L2 regularization
- L2 regularization
- Dropout
- Batch normalization
- Learning-rate decay
- SGD with momentum
- Adam
- Data augmentation combinations
- Deeper VGG-style configurations
- Alternative epoch, batch-size, filter, and dropout settings

### 04. CIFAR-10 Data Augmentation
`04_cifar10_data_augmentation.ipynb`

**Coverage**
- ImageDataGenerator
- Horizontal and vertical flips
- Translation and shifts
- Rotation
- Zoom
- Brightness
- Shear
- RandomFlip
- RandomRotation
- RandomZoom
- RandomCrop
- RandomTranslation
- RandomContrast
- RandomBrightness
- RandAugment
- Combined augmentation layers
- Mirrored and augmented test-time prediction

### 05. Flower Image Classification with tf.data and CNN Augmentation
`05_flower_tfdata_cnn_augmentation.ipynb`

**Coverage**
- TensorFlow flower dataset
- Directory-based image loading
- Training/validation split
- Image batches and class labels
- `tf.data`
- Cache and prefetch
- Multiple Keras augmentation layers
- CNNs with integrated augmentation

### 06. CIFAR-10 Advanced CNNs and Pretrained Models
`06_cifar10_pretrained_cnn_models.ipynb`

**Coverage**
- VGG-style CNN blocks
- Model checkpoints
- Early stopping
- ReduceLROnPlateau
- Custom termination callback
- ImageDataGenerator
- Parallel convolutional branches
- Functional API
- ResNet50 pretrained on ImageNet
- Custom classification head
- Test-time augmentation
- Multi-view prediction aggregation

## Repository Structure

```text
4. Convolutional Neural Networks (CNNs)/
│
├── README.md
├── requirements.txt
├── .gitignore
├── 01_mnist_fashion_mnist_cifar10_cnn_fundamentals.ipynb
├── 02_cifar10_cnn_architecture_design.ipynb
├── 03_cifar10_cnn_regularization_optimization.ipynb
├── 04_cifar10_data_augmentation.ipynb
├── 05_flower_tfdata_cnn_augmentation.ipynb
└── 06_cifar10_pretrained_cnn_models.ipynb
```

## Environment

```bash
pip install -r requirements.txt
```

MNIST, Fashion-MNIST, CIFAR-10, and the TensorFlow flower dataset are loaded directly by the
notebooks. The standalone image-augmentation demonstration asks the user to upload an image from
their computer when run in Google Colab.

## Next Portfolio Section

The next section can focus on **Dense vs. Convolutional Networks**, using the architecture and
image-classification results established here as the basis for direct comparison.
