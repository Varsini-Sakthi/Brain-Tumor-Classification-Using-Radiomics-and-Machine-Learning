# Brain Tumor Classification Using Radiomics and Machine Learning
This project implements an end-to-end radiomics-based machine learning pipeline for brain tumor classification using MRI images. Hand-crafted texture and intensity features are extracted from tumor regions and used to train and evaluate multiple machine learning models.

The workflow covers image preprocessing, feature extraction, model training, evaluation, and comparison, making it suitable for research and educational purposes in medical imaging, bioinformatics, and applied machine learning.

# Project Overview
Input: Brain MRI images with tumor masks (.mat files)

1. Feature Engineering:

* First-order statistical features
* GLCM texture features

2. Models Implemented:

* Logistic Regression
* Random Forest
* XGBoost

3. Evaluation Metrics:

* Accuracy
* Precision, Recall, F1-score (macro)
* ROC-AUC (One-vs-Rest)

# Repository Structure
```bash
├── BrainTumor.ipynb # Complete pipeline notebook
├── radiomics_features.csv # Extracted radiomics features 
├── dataset/
│   └── data/ # MRI .mat files 
├── README.md # Project documentation 

```
# Dataset Description

Each .mat file contains:
* MRI image
* Tumor mask
* Tumor label (multi-class)
* Patient ID (PID)
Tumor masks are used to isolate the region of interest for radiomics feature extraction.

# Methodology
1. Preprocessing
* Image normalization (z-score)
* Tumor region isolation using binary masks
* Visualization of MRI images with overlayed tumor masks

2. Feature Extraction

First-Order Statistics:
* Mean
* Standard Deviation
* Minimum & Maximum
* Skewness
* Kurtosis

Texture Features (GLCM):
* Contrast
* Dissimilarity
* Homogeneity
* Energy
* Correlation

All features are computed only within the tumor region.

3. Machine Learning Models

| Model | Description | 
| :--- | :--- | 
| Logistic Regression | Baseline linear classifier with feature scaling | 
| Random Forest | Ensemble tree-based model handling non-linearities | 
| XGBoost | Gradient-boosted trees optimized for multi-class classification |

Class imbalance is handled using balanced class weights where applicable.

# Evaluation
* Stratified train-test split
* Macro-averaged metrics for multi-class robustness
* ROC-AUC computed using One-vs-Rest strategy
* Performance comparison across models

# Technologies Used

* Python
* NumPy, Pandas
* scikit-image
* scikit-learn
* XGBoost
* Matplotlib
* SHAP (model interpretability)

# How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-radiomics-ml.git
cd brain-tumor-radiomics-ml
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place the dataset in:
```bash
dataset/data/
```
4. Run the notebook:
```bash
jupyter notebook BrainTumor.ipynb
```
# Future Improvements

* Deep learning feature extraction (CNN-based radiomics)
* Cross-validation with patient-wise splitting
* External dataset validation
* Explainability with SHAP/LIME visualizations
* Clinical outcome prediction integration

# Author
Varsini Sakthivadivel Ramasamy

MS Bioinformatics

Johns Hopkins University
