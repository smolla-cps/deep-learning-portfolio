# CelebA Data

The first three notebooks download their datasets through TensorFlow or scikit-learn.

The CelebA notebook expects the following files:

```text
Datasets/
├── img_align_half_res.zip
├── image_class.xlsx
└── list_attr_celeba.csv
```

- `img_align_half_res.zip` contains the CelebA images used by the notebook.
- `image_class.xlsx` contains the binary-class experiment labels used in the single-attribute sections.
- `list_attr_celeba.csv` contains the standard multi-attribute labels for the 40-attribute experiments.

Download the dataset from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and upload from the desktop. Alternate option is: in Google Colab, place the files in Google Drive and update the `DATA_ROOT` variable in `04_celeba_attribute_classification.ipynb` if your folder differs from:

```python
DATA_ROOT = Path("/content/drive/MyDrive/Datasets")
```

The notebook extracts the image archive into `celebA_imgs/` at runtime.
