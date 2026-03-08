# Age Detection from Facial Images
This project explores various machine learning concepts:unsupervised (K-Means), supervised deep learning (CNN), and classical supervised learning (SVM) to estimate and categorize human age from facial images.

## Project Overview
Age estimation is a complex computer vision task due to significant variations in how aging manifests across individuals. This project implements multiple models to compare their effectiveness in extracting age-relevant features and explores the transition from fine-grained classification to broader age categories.

## Dataset
* **Source:** [Faces: Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset/data) from Kaggle.
* **Scale:** 20,000+ images with individuals ranging from 0 to 116 years old.
* **Demographics:** Diverse ethnic backgrounds including White, Black, Asian, and Indian.
* **Preprocessing:** Images were resized to $128 \times 128$ pixels and normalized. Precise numerical ages were extracted from filenames for granular analysis.

## Models & Methodology

### 1. K-Means Clustering (Unsupervised)
* **Approach:** Implemented with 10 clusters after reducing dimensionality via PCA (50 components).
* **Key Finding:** Unsupervised clustering demonstrated that facial images don't naturally group by age alone; other visual features like lighting, pose, and background often dominate the process.

### 2. Convolutional Neural Networks (CNN)
* **Custom Architecture:** Built a multi-block CNN featuring Batch Normalization and Dropout for regularization.
* **Transfer Learning:** Leveraged **VGG16** (pre-trained on ImageNet).
* **Optimization:** Used fine-tuning by unfreezing top layers with a very low learning rate ($1e-5$) to adapt specifically to facial aging patterns.

### 3. Support Vector Machine (SVM)
* **Approach:** Used an RBF kernel on PCA-reduced features.
* **Performance:** While it showed bias toward well-represented age groups (20-39), it performed effectively on broader age categories.

## Key Results
The study found that appropriate task framing (grouping ages) significantly improved model reliability.

| Model | Task | Accuracy |
| :--- | :--- | :--- |
| **Fine-Tuned VGG16** | 11 Age Categories | ~44.55% |
| **Custom CNN** | 11 Age Categories | ~38.25% |
| **Optimized SVM** | 5 Broad Categories | ~57.00% |
| **VGG16 (Grouped)** | 5 Broad Categories | ~56.68% |

## Technologies Used
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-learn
* **Computer Vision:** OpenCV, PIL
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## Installation
```bash
pip install tensorflow scikit-learn opencv-python seaborn pillow tqdm
