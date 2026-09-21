# Cherry Leaf Disease Classification with CNN

Classifying cherry leaf images into 5 categories (healthy and four diseases) with Convolutional Neural Networks (Keras): EDA, preprocessing, a manually built AlexNet baseline, an improved custom CNN, and evaluation on a held-out test set. Built as a Deep Learning course assignment.

## Dataset

3,642 leaf images across 5 classes, all 800 × 1000 px (aspect ratio 0.8). The images are not included in this repository.

| Class | Images |
|-------|--------|
| Cherry Leaf Scorch | 1,101 |
| Cherry purple leaf spot | 987 |
| Cherry brown_spot | 614 |
| Cherry Normal leaf | 500 |
| Cherry_shot hole disease | 440 |

The dataset is imbalanced (largest vs. smallest class: 661 images).

## Preprocessing

- Per-class 70:15:15 train/validation/test split
- Resize to 224 × 224 and rescale pixels to [0, 1]
- Augmentation on the training set only: rotation (20°), zoom (0.2), horizontal flip
- `class_weight='balanced'` to compensate for class imbalance

## Models

| | Architecture | Parameters | Training |
|---|---|---|---|
| Baseline | AlexNet built manually: 5 Conv layers + 2 Dense (4096) + softmax | 46.8M | Adam, 10 epochs |
| Improved | Custom CNN: 4 × (Conv + BatchNorm + MaxPool), Dense 256 / 128 with Dropout (0.5 / 0.3) | 13.3M | Adam (lr 1e-4), 20 epochs, EarlyStopping (patience 4), ReduceLROnPlateau (factor 0.3, patience 2) |

## Results (test set)

| Model | Accuracy | Loss |
|-------|----------|------|
| AlexNet (baseline) | 82.50% | 0.4785 |
| Custom CNN (improved) | 95.83% | 0.1126 |

Per-class report for the baseline (precision / recall / F1):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cherry Leaf Scorch | 0.94 | 0.81 | 0.87 |
| Cherry Normal leaf | 0.71 | 0.66 | 0.68 |
| Cherry brown_spot | 0.73 | 0.84 | 0.78 |
| Cherry purple leaf spot | 0.94 | 0.93 | 0.94 |
| Cherry_shot hole disease | 0.65 | 0.79 | 0.71 |

Macro F1 0.80, weighted F1 0.83.

## Notes

- The split folders found by the generators contain 3,293 / 997 / 1,006 images (5,296 in total), more than the 3,642 source images. Repeated runs of the split cell (random shuffle without a seed, copying into existing folders) can put the same image into more than one split, which would inflate the test results. The split should be regenerated from an empty `dataset_split/` folder with a fixed seed, and the models re-evaluated.
- The per-class classification report covers the baseline model only; the improved model is evaluated on accuracy and loss.
- The notebook was run on Google Colab with the dataset on Google Drive (`base_path` and `drive.mount`); adjust these paths to run it elsewhere.
