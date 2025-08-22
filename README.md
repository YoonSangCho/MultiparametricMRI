# MultiparametricMRI
This repository contains the implementation of machine learning (ML) experiments for predicting germline BRCA1/2 mutations in high-risk breast cancer patients using multiparametric MRI (mpMRI) features.
The study integrates clinical, morphologic, kinetic (CAD-derived), and diffusion-weighted (ADC) features into ML-based prediction models. We conducted systematic experiments with multiple models, feature sets, and repeated validation to ensure robustness and reproducibility.

## Experiment Design

1. Data Source
  - Retrospective dataset of 231 high-risk breast cancer patients (82 BRCA+, 149 BRCA−).
  - mpMRI features include:
    - Clinical: Age, histologic grade, tumor size, subtype, etc.
    - Morphologic: Background parenchymal enhancement (BPE), peritumoral edema, axillary adenopathy, etc.
    - Kinetic (CAD-derived): Tumor size, angio-volume, peak enhancement, washout component, etc.
    - ADC values: Derived from DWI sequences.
2. Preprocessing
  - Train/test split (80%/20%) with stratification.
  - Z-score normalization applied to continuous variables.
  - Random seeds fixed for reproducibility.

3. Models Compared
  - Logistic Regression, Elastic Net, LDA, QDA
  - k-NN, Naive Bayes, MLP (Neural Network)
  - Decision Tree, Random Forest, Extra Trees, AdaBoost
  - XGBoost, LightGBM, CatBoost, SVM (RBF)

4. Evaluation
  - Hyperparameter tuning via GridSearchCV with 5-fold Stratified Cross-Validation.
  - ROC-AUC used as the primary metric.
  - 30 repeated runs with different random seeds.
