import os, random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# optional libs
HAS_XGB = HAS_LGB = HAS_CAT = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGB = False
try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CAT = False

def _pos_weight_from_y(y):
    y = np.asarray(y).ravel()
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    return n_neg / n_pos  


def build_model_specs(y):
    spw = float(_pos_weight_from_y(y))  # e.g., 149/82 ≈ 1.82

    MODEL_SPECS = {}

    # 1) Logistic (L1/L2)
    MODEL_SPECS["logistic_regression"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="saga"))
        ]),
        {
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__max_iter": [1000, 3000],
            "clf__tol": [1e-3, 1e-4],
            "clf__class_weight": [None, "balanced"],
        }
    )

    # 2) Elastic Net Logistic
    MODEL_SPECS["logistic_elasticnet"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="saga", penalty="elasticnet"))
        ]),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.25, 0.5, 0.75],
            "clf__max_iter": [2000, 5000],
            "clf__tol": [1e-3, 1e-4],
            "clf__class_weight": [None, "balanced"],
        }
    )

    # 3) Decision Tree 
    MODEL_SPECS["decision_tree"] = (
        DecisionTreeClassifier(),
        {
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5]
        }
    )

    # 4) Random Forest
    MODEL_SPECS["random_forest"] = (
        RandomForestClassifier(n_jobs=-1, n_estimators=400),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }
    )

    # 5) Extra Trees
    MODEL_SPECS["extra_trees"] = (
        ExtraTreesClassifier(n_jobs=-1, n_estimators=400),
        {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }
    )

    # 6) AdaBoost
    MODEL_SPECS["adaboost"] = (
        AdaBoostClassifier(),
        {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.5, 1.0]
        }
    )

    # 7) SVM (RBF)
    MODEL_SPECS["svm_rbf"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True))
        ]),
        {
            "clf__C": [0.5, 1.0, 5.0, 10.0],
            "clf__gamma": ["scale", 0.1, 0.01],
            "clf__class_weight": [None, "balanced"],
        }
    )

    # 8) k-NN
    MODEL_SPECS["knn"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        {
            "clf__n_neighbors": [3, 5, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
    )

    # 9) Naive Bayes (Gaussian)
    MODEL_SPECS["naive_bayes"] = (
        GaussianNB(),
        {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    )

    # 10) MLP
    MODEL_SPECS["mlp"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=1000))
        ]),
        {
            "clf__hidden_layer_sizes": [(64,), (64, 32)],
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3, 1e-2],
            "clf__early_stopping": [True],
        }
    )

    # 11) LDA  (scaler on/off 동시 탐색)
    MODEL_SPECS["lda"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis())
        ]),
        [
            {"scaler": [StandardScaler(), "passthrough"], "clf__solver": ["svd"]},
            {"scaler": [StandardScaler(), "passthrough"], "clf__solver": ["lsqr"], "clf__shrinkage": ["auto", 0.1, 0.5, 0.9]},
        ]
    )

    # 12) QDA (scaler on/off 동시 탐색)
    MODEL_SPECS["qda"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", QuadraticDiscriminantAnalysis())
        ]),
        {
            "scaler": [StandardScaler(), "passthrough"],
            "clf__reg_param": [0.0, 0.1, 0.2],
        }
    )

    # 13) CatBoost
    if HAS_CAT:
        # class_weights는 [w0, w1] 형태 (합이 1일 필요는 없음)
        w0, w1 = 1.0, spw  # 간단히 양성 가중만 키우는 방식
        MODEL_SPECS["catboost"] = (
            CatBoostClassifier(
                iterations=700, loss_function="Logloss", verbose=False
            ),
            {
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5],
                # 불균형 보정: class_weights 후보
                "class_weights": [[1.0, 1.0], [w0, w1]],
            }
        )

    return MODEL_SPECS
