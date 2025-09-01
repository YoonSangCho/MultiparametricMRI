# MultiparametricMRI
This repository contains the  official codes for the implementation of machine learning (ML) experiments in the predictions of germline BRCA1/2 mutations of high-risk breast cancer patients using multiparametric MRI (mpMRI) features. The study investigated integrates clinical, morphologic, kinetic (CAD-derived), and diffusion-weighted (ADC) features into ML-based prediction models. We conducted systematic experiments with multiple models, feature sets, and repeated validation to ensure robustness and reproducibility.

### Citation
Park H, Cho KR, Lee S, Cho D, Park KH, Cho YS, Song SE. Prediction of Germline BRCA Mutations in High-Risk Breast Cancer Patients Using Machine Learning with Multiparametric Breast MRI Features (Sensors, Sep. 2025)

### How to Run
**1. create enviromnent**
- conda create -n brca-mri python=3.10
- conda activate brca-mri
- pip install -r requirements.txt

**2. Run Experiments**
- python main.py


### Experiment Design

**1. Data Source**
  - Retrospective dataset of 231 high-risk breast cancer patients (82 BRCA+, 149 BRCA−).
  - 1) Clinical & Pathological: Age, histologic grade, mean invasive tumor size, Axillary lymph node metastasis, subtype, etc.
  - 2) Multiparametric MRI (mpMRI) features include:
    - (1) Morphologic: Background parenchymal enhancement (BPE), peritumoral edema, axillary adenopathy, etc.
    - (2) Kinetic (CAD-derived): Tumor size, angio-volume, peak enhancement, washout component, etc.
    - (3) ADC (DWI-derived): mean, minimum, maximum, standard deviation of ADC values.

**2. Preprocessing**
  - Train/test split (80%/20%) with stratification.
  - Z-score normalization applied to continuous variables.
  - Random seeds fixed for reproducibility.

**3. Models Compared**
  - Logistic Regression, Elastic Net, SVM (RBF)
  - Naive Bayes, Linear Disciminant Analysis, Quadratic Discriminant Analysis (QDA) 
  - k-NN, , MLP (Neural Network)
  - Decision Tree, Random Forest, Extra Trees, AdaBoost XGBoost,

**4. Evaluation**
  - 30 repeated runs with different random seeds.
  - Hyperparameter tuning via GridSearchCV with 5-fold Stratified Cross-Validation.
  - ROC-AUC used as the primary metric.
  

