# 🖼️ Caltech-101 Image Classification Pipeline

## 📌 Overview
This repository contains a complete, memory-optimized Machine Learning pipeline for classifying images from the **Caltech-101** dataset. The project demonstrates a robust, classic Computer Vision approach, leveraging **Histogram of Oriented Gradients (HOG)** and **Support Vector Machines (SVM)**, without relying on Deep Learning architectures.

## ✨ Key Features
* **Exploratory Data Analysis (EDA):** Comprehensive analysis of class distributions, aspect ratios, and color channels using `Seaborn` and `Pandas`.
* **Memory-Optimized Preprocessing:** Efficient image resizing and loading using `OpenCV` and Python's Garbage Collector (`gc`) to prevent RAM overflow during matrix stacking.
* **Custom Data Augmentation:** Addressing class imbalance by synthetically generating images (rotation, flipping, brightness adjustment) up to the 75th percentile of the class distribution.
* **Feature Extraction:** Advanced dimensionality reduction from raw pixels using `skimage.feature.hog` followed by `PCA` (Principal Component Analysis).
* **Hyperparameter Tuning:** Efficient model selection and optimization using `RandomizedSearchCV` on a Support Vector Classifier.
* **Comprehensive Error Analysis:** Visualizing confusion matrices and extracting the most misclassified pairs to evaluate system blind spots.

## 📊 Model Performance
The model was evaluated on a completely unseen, un-augmented test set (20% of the dataset) with the following strictly verified metrics:

* **Test Accuracy:** `66.31%`
* **F1-Score (Weighted):** `66.55%`
* **F1-Score (Macro):** `51.86%`

> *Note: Achieving >66% accuracy on a highly complex dataset of 101 diverse classes using strictly classical Machine Learning techniques (SVM + HOG) indicates a highly robust feature extraction pipeline.*

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, Scikit-Image
* **Machine Learning:** Scikit-Learn
* **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn

## 🚀 How to Run
1. Clone this repository to your local machine.
2. Download the `caltech101` dataset and place the folder directly in the root directory.
3. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image joblib