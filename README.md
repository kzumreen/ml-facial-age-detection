# Age Detection from Facial Images
Implemented and compared three machine learning approaches K-Means clustering, 
custom CNN, and VGG16 transfer learning for facial age estimation across 20,000+ 
images spanning 11 age categories. Achieved 57% accuracy on 5-class grouping and 
demonstrated that task framing significantly impacts model performance.

![Sample Age Group Predictions](images/sample_predictions.png)
*Sample predictions from the fine-tuned VGG16 model performs strongest on 21-30 
(most represented class) and shows adjacent-class confusion on harder age groups*

---

## Project Overview

Age estimation is a complex computer vision task due to significant variation in 
how aging manifests across individuals. This project implements and compares 
multiple approaches from unsupervised clustering to transfer learning to 
understand both the capability and limits of each method on the same dataset.

A key finding emerged through iteration: reframing the problem from 11 granular 
age categories to 5 broader demographic groups significantly improved model 
reliability across all approaches, pointing to a fundamental challenge of 
inter-class similarity in facial aging.

---

## Dataset

- **Source:** [Faces: Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset/data) from Kaggle
- **Scale:** 20,000+ images, ages 0–116
- **Demographics:** Diverse ethnic backgrounds including White, Black, Asian, and Indian
- **Preprocessing:** Resized to 128×128, normalized. Precise ages extracted from filenames for granular analysis

![Age Group Distribution](images/age_distribution.png)
*Significant class imbalance — 21-30 group contains ~1,150 images vs fewer than 
50 for ages 1-10 and 101+, directly affecting model bias toward majority classes*

---

## Models & Methodology

### 1. K-Means Clustering (Unsupervised)

- Dimensionality reduced via PCA (50 components) before clustering
- Tested k=2 through k=15 using silhouette scores — k=2 achieved highest 
  cohesion (0.20) but k=10 chosen for age-relevant granularity (score: 0.0858)

![K-Means PCA Projection](images/kmeans_pca.png)
*2D PCA projection of 10 K-Means clusters — well-separated spatially but 
clusters capture visual similarity patterns rather than chronological age*

**Key finding:** Unsupervised clustering confirmed that facial images don't 
naturally group by age. Lighting, pose, and background often dominate over 
age-related features — motivating the move to supervised methods.

---

### 2. Convolutional Neural Networks (CNN)

- **Custom CNN:** Multi-block architecture with Batch Normalization and Dropout 
  for regularization
- **Transfer Learning:** VGG16 pre-trained on ImageNet, fine-tuned by unfreezing 
  top layers at learning rate 1e-5 to adapt to facial aging patterns
- Training accuracy reached ~91% while validation plateaued at ~40%, confirming 
  overfitting — addressed via dropout and learning rate scheduling

![CNN Training Curves](images/training_curves.png)
*Training vs validation accuracy and loss over epochs — visible overfitting gap 
motivates architectural improvements and broader age grouping*

![CNN Confusion Matrix](images/cnn_confusion_matrix.png)
*Confusion matrix showing strong bias toward 21-30 class — directly reflects 
class imbalance in training data*

---

### 3. Support Vector Machine (SVM)

- RBF kernel on PCA-reduced image features
- Tested with both 11 granular and 5 broad age categories
- Class weighting applied to address imbalance

![SVM Confusion Matrix](images/svm_confusion_matrix.png)
*SVM with 5 broad age groups — improvement over 11-class version, but still 
biased toward the 20-39 majority class*

---

## Key Results

| Model | Task | Accuracy |
|---|---|---|
| Fine-Tuned VGG16 | 11 age categories | ~44.55% |
| Custom CNN | 11 age categories | ~38.25% |
| VGG16 (grouped) | 5 broad categories | ~56.68% |
| Optimized SVM | 5 broad categories | ~57.00% |

The consistent ~15% accuracy gain from switching to 5 broad categories across 
all models confirms that the performance ceiling for 11-class age estimation is 
fundamentally limited by inter-class similarity, not model architecture.

---

## Technologies Used

**Deep Learning:** TensorFlow / Keras  
**Machine Learning:** Scikit-learn  
**Computer Vision:** OpenCV, PIL  
**Data Handling:** Pandas, NumPy  
**Visualization:** Matplotlib, Seaborn

---

## How to Navigate this Repo

- `notebook.ipynb` — Full pipeline: EDA, preprocessing, K-Means, CNN, VGG16, SVM
- `report/` — Project report with full methodology and analysis
