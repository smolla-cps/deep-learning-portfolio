# Transfer Learning

This portfolio section demonstrates how pretrained convolutional networks can be adapted to new image-classification tasks.

The notebooks progress from the standard two-stage transfer-learning workflow to a broader set of experiments involving freeze depth, classifier design, initialization, and pretrained-backbone selection.

## Repository Structure

```text
5. Transfer Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_cats_dogs_mobilenetv2_transfer_learning.ipynb
└── 02_flower_transfer_learning_experiments.ipynb
```

## Notebook 1: Cats vs Dogs with MobileNetV2

Introduces the fundamental transfer-learning workflow:

- Cats vs Dogs dataset preparation from the official Microsoft dataset
- ImageNet-pretrained MobileNetV2
- removal of the original ImageNet classifier
- frozen feature extraction
- image augmentation
- MobileNetV2 preprocessing
- global average pooling
- task-specific binary classification head
- frozen-backbone training
- selective unfreezing
- partial fine-tuning
- reduced learning rate during fine-tuning
- training and validation metric analysis

## Notebook 2: Flower Transfer-Learning Experiments

Extends the workflow to a five-class flower dataset and investigates:

- standard train/validation splitting
- MobileNetV2 transfer learning
- freeze-depth sweep
- fully frozen and partially fine-tuned endpoints
- validation-based freeze-depth selection
- larger task-specific classification head
- batch normalization and dropout
- custom classification-layer initialization
- label inspection
- alternative pretrained backbone with InceptionV3
- multiclass softmax classification

## Technical Corrections Incorporated

The notebooks use runtime-safe dataset extraction paths, explicit max/min metric reporting, independent pretrained models for comparison experiments, correct application of the selected freeze depth, and a softmax output with sparse categorical crossentropy for the five-class flower task.

## Datasets

Both datasets are downloaded automatically:

- **Cats vs Dogs**: binary image classification using the official Microsoft Kaggle Cats and Dogs archive
- **Flower Photos**: five-class flower classification

No dataset files need to be stored in the repository.

## Portfolio Progression

```text
CNN Fundamentals
→ Advanced / Non-Sequential Architectures
→ Transfer Learning
→ Autoencoders
→ Image Segmentation
```

Transfer learning connects naturally to later segmentation models because pretrained CNN backbones can also be reused as feature encoders in encoder-decoder architectures.


## Dataset Download Note

The Cats vs Dogs notebook uses the official Microsoft Kaggle Cats and Dogs archive rather than the older filtered Google Storage archive, which can return HTTP 403 in current Colab runtimes. The notebook validates images and constructs a balanced 3,000-image subset for the transfer-learning workflow.

## Running the Notebooks

Google Colab with GPU acceleration is recommended. Run each notebook from top to bottom in a fresh runtime.
