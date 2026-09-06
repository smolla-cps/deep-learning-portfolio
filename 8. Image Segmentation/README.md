# Image Segmentation with U-Net and MobileNetV2

This portfolio project implements semantic image segmentation on the **Oxford-IIIT Pet** dataset using a modified **U-Net** architecture with a pretrained **MobileNetV2** encoder.

The notebook progresses from dataset preparation and synchronized image-mask augmentation to transfer-learning architecture design, training, qualitative prediction analysis, and a short demonstration of per-pixel sample weighting for class imbalance.

## Technical Highlights

- Oxford-IIIT Pet images and trimap masks downloaded from the official Oxford source
- Official `trainval.txt` and `test.txt` dataset partitions
- Image and segmentation-mask preprocessing at `128 × 128`
- Synchronized image-mask augmentation
- Efficient `tf.data` input pipeline
- Pretrained MobileNetV2 encoder
- Modified U-Net decoder with skip connections
- Sparse multiclass pixel classification
- Training and validation loss visualization
- Qualitative segmentation predictions
- Per-pixel sample-weight construction for class imbalance

## Repository Structure

```text
8. Image Segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── 01_oxford_iiit_pet_unet_segmentation.ipynb
```

## Dataset

The **Oxford-IIIT Pet** dataset contains images of 37 pet breeds with pixel-level trimap segmentation annotations. The notebook downloads the official image and annotation archives automatically, so dataset files are not stored in this repository.

The trimap masks contain three pixel classes. During preprocessing, mask labels are shifted from `{1, 2, 3}` to `{0, 1, 2}` for sparse categorical training.

## Model

The segmentation model follows a modified U-Net structure:

1. **Encoder:** pretrained MobileNetV2 extracts multiscale feature maps.
2. **Skip connections:** intermediate encoder activations preserve spatial detail.
3. **Decoder:** transposed-convolution blocks progressively restore spatial resolution.
4. **Output:** three logits are produced for every pixel.

The MobileNetV2 encoder is frozen during the demonstrated training workflow.

## Class-Imbalance Demonstration

The main U-Net is trained using the standard segmentation loss. A second model demonstrates how **per-pixel sample weights** can be supplied to Keras when different pixel classes need different emphasis.

The weighted section is intentionally short and is included to demonstrate the technique rather than claim a performance improvement. A fair comparison would require the same training schedule for both models and segmentation-specific metrics such as mean IoU or Dice score.

## Running the Notebook

The notebook is designed for Google Colab or another TensorFlow environment with internet access.

1. Open `01_oxford_iiit_pet_unet_segmentation.ipynb`.
2. Run the cells from top to bottom.
3. The Oxford-IIIT Pet dataset will be downloaded automatically.
4. GPU acceleration is recommended for model training.

## Requirements

See `requirements.txt` for the primary Python packages.

## Attribution

This project is adapted from concepts and code patterns in the TensorFlow image-segmentation tutorial. TensorFlow-originated material is retained with its Apache 2.0 attribution.
