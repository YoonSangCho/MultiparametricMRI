import os
import random
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.base import clone
from models_specs import build_model_specs

def load_data(data_type="clinical"):
    # Load the dataset
    df = pd.read_excel("./data/data_BRCA.xlsx")
    df['Histologic grade'] = df['Histologic grade'].astype(int).replace({1: 0, 2: 0, 3: 1}) #, 4: 1
    # Define column names for different data types
    colnames_clinical = ['age.1', 'Histologic grade', 'invasive tumor size', 'luminal'] #'SUBTYPE-luminal', 
    colnames_kinetic = ['Pre-NAC CADTumor size', 'CAD angiovolume', 'CAD peak enhancement', 'CAD delayed phase (Persistent)', 'CAD delayed phase (washout)']
    colnames_morphologic = ['BPE_2_category', 'T2WI Peritumoral edema (0=No, 1=Yes)', 'LN enlargement']
    colnames_multivariate = [
        'BPE_2_category', 
        'Pre-NAC CADTumor size', 
        'CAD delayed phase (washout)'
        ]
    colnames_univariate = [
        'BPE_2_category', 'T2WI Peritumoral edema (0=No, 1=Yes)', 'LN enlargement', 
        'Pre-NAC CADTumor size', 'CAD angiovolume', 'CAD peak enhancement', 'CAD delayed phase (Persistent)', 'CAD delayed phase (washout)'        
        ]
    
    # (1) Clinical
    if data_type=="clinical": 
        X = df[colnames_clinical]
    # (2) Kinetic 
    elif data_type == "kinetic": 
        X = df[colnames_kinetic]
    # (3) morphologic variables in Table 2
    elif data_type == "morphologic": # Morphologic features
        X = df[colnames_morphologic]
    # (4) Clinical + Kinetic  
    elif data_type == "clinic-kinetic": 
        X = df[colnames_clinical + colnames_kinetic]
    # (5) Clinicial + morphologic
    elif data_type == "clinic-morphologic": 
        X = df[colnames_clinical + colnames_morphologic]
    # (6) Kinetic + Morphologic = Univariate features
    elif data_type == "kinetic-morphologic": 
        X = df[colnames_kinetic + colnames_morphologic]
    # (7) Clinical + Kinetic + Morphologic 
    elif data_type == "clinic-kinetic-morphologic": 
        X = df[colnames_clinical + colnames_kinetic + colnames_morphologic]
    # (8) Multivariate features
    elif data_type == "multivariate": # Multivariate features
        X = df[colnames_multivariate]
    # (9) clinical + multivariate features
    elif data_type == "clinic-multivariate": 
        X = df[colnames_clinical + colnames_multivariate]
    else:
        raise ValueError(f"지원하지 않는 data_type: {data_type}")
    
    # one hot encoding for 'SUBTYPE-luminal' and 'luminal'
    if 'luminal' in X.columns:
        X = pd.get_dummies(X, columns=['luminal'], drop_first=True)   
    
    X = X.values.astype(float)
    y = df['BRCA 1 or 2'].values.astype(float).ravel()
    return X, y

def set_reproducible_seed(estimator, seed: int):
    """
    Pipeline이면 clf 스텝에, 아니면 바로 추정기에 random_state/random_seed를 세팅.
    존재하지 않으면 조용히 패스.
    """
    try:
        if isinstance(estimator, Pipeline):
            if "clf" in estimator.named_steps:
                clf = estimator.named_steps["clf"]
                clf_params = clf.get_params(deep=False)
                if "random_state" in clf_params:
                    estimator.set_params(**{"clf__random_state": seed})
                elif "random_seed" in clf_params:  # CatBoost 구버전 호환
                    estimator.set_params(**{"clf__random_seed": seed})
        else:
            params = estimator.get_params(deep=False)
            if "random_state" in params:
                estimator.set_params(random_state=seed)
            elif "random_seed" in params:  # CatBoost
                estimator.set_params(random_seed=seed)
    except Exception:
        pass
    return estimator

def run_experiment(X, y, n_runs=30, save_dir="./results/"):
    save_dir_all = save_dir + "all_runs"
    os.makedirs(save_dir_all, exist_ok=True)
    results_rows, pred_rows = [], []

    for seed in range(n_runs):
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        MODEL_SPECS = build_model_specs(y_tr)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for model_name, (pipe, param_grid) in MODEL_SPECS.items():
            print(f"\n=== Model: {model_name} ===")
            estimator = clone(pipe)
            estimator = set_reproducible_seed(estimator, seed)

            grid = GridSearchCV(estimator=estimator, param_grid=param_grid,
                                scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0)
            grid.fit(X_tr, y_tr)
            best_est = grid.best_estimator_

            if hasattr(best_est, "predict_proba"):
                y_score = best_est.predict_proba(X_te)[:, 1]
            elif hasattr(best_est, "decision_function"):
                s = best_est.decision_function(X_te)
                y_score = s[:, 1] if (isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == 2) else np.ravel(s)
            else:
                y_score = best_est.predict(X_te)
            y_pred = (y_score >= 0.5).astype(int) if np.ndim(y_score) == 1 else best_est.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            results_rows.append({
                "seed": seed, "model": model_name, "best_params": grid.best_params_,
                "cv_best_auc": grid.best_score_,
                "accuracy": accuracy_score(y_te, y_pred),
                "sensitivity": recall_score(y_te, y_pred),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                "f1": f1_score(y_te, y_pred),
                "auc": roc_auc_score(y_te, y_score) if len(np.unique(y_te)) == 2 else np.nan
            })

            pred_rows.append(pd.DataFrame({
                "seed": seed, "model": model_name, "y_test": y_te,
                "y_pred": y_pred, "y_score": y_score
            }))

            pd.DataFrame(classification_report(y_te, y_pred, output_dict=True)).to_csv(
                f"{save_dir_all}/classification_report_{model_name}_seed{seed}.csv", encoding="utf-8-sig"
            )

    df_metrics = pd.DataFrame(results_rows)
    df_preds = pd.concat(pred_rows, axis=0).reset_index(drop=True)
    df_metrics.to_csv(f"{save_dir}/summary_metrics_all_runs.csv", index=False, encoding="utf-8-sig")
    df_preds.to_csv(f"{save_dir}/predictions_all_runs.csv", index=False, encoding="utf-8-sig")
    print(df_metrics.groupby("model")[["auc", "accuracy", "sensitivity", "specificity", "f1"]].mean().sort_values("auc", ascending=False))


if __name__ == "__main__":
    ## 추후 argparse나 config 파일 활용
    import os
    import sys
    from models_specs import build_model_specs  # Import the function to build model specifications

    os.chdir("/mnt/workspace/workspace_multiMRI")
    print("=== MultiMRI Experiment ===")
    for data_type in ["clinical", "kinetic", "morphologic", 
                      "clinic-kinetic", "clinic-morphologic", 
                      "kinetic-morphologic", "clinic-kinetic-morphologic", 
                      "multivariate", "clinic-multivariate"
                      ]:
        print(f"=== Data Type: {data_type} ===")
        X, y = load_data(data_type=data_type)
        os.makedirs(f"./results_{data_type}", exist_ok=True)
        save_dir = f"./results_{data_type}/"
        run_experiment(X, y, n_runs=30, save_dir=save_dir)

