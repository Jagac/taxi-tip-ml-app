import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_parquet(
    "/home/jagac/projects/taxi-tip-mlapp/Research/yellow_tripdata_2023-04.parquet"
)
df.columns

df["tipped"] = (df["tip_amount"] > 0).astype("int")
df["tipped"].value_counts()

df["trip_time"] = (df.tpep_pickup_datetime - df.tpep_dropoff_datetime).astype(
    "timedelta64[s]"
) / np.timedelta64(1, "s")
df = df.drop(["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount"], axis=1)

one_hot_enc = OneHotEncoder()
arr = one_hot_enc.fit_transform(df[["store_and_fwd_flag"]])
store_and_fwd_flag = pd.DataFrame(arr, columns=["store_and_fwd_flag_ohe"])

df_merge = pd.merge(df, store_and_fwd_flag, left_index=True, right_index=True)
df_merge = df_merge.reset_index(drop=True)
df_merge = df_merge.drop("store_and_fwd_flag", axis=1)
df_merge


y = df_merge.tipped
X = df_merge.drop("tipped", axis=1)
X = df_merge.drop("store_and_fwd_flag_ohe", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


oversample = RandomOverSampler(sampling_strategy="all")
X_over, y_over = oversample.fit_resample(X_train, y_train)

print(X_train.shape)
print(y_train.shape)
print(X_over.shape)
print(y_over.shape)

import mlflow
from pathlib import Path

# Set tracking URI
MODEL_REGISTRY = Path("mlruns")
Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))


import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def objective(trial):
    """Define the objective function"""
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
        "eval_metric": "mlogloss",
    }

    optuna_model = XGBClassifier(**params)
    optuna_model.fit(X_over, y_over)

    y_pred = optuna_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


from optuna.integration.mlflow import MLflowCallback

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(
    study_name="xgboost_optimization", direction="maximize", pruner=pruner
)
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), metric_name="accuracy"
)

study.optimize(objective, n_trials=2, callbacks=[mlflow_callback])


import json

print(f"Best value (f1): {study.best_trial.value}")
print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

import mlflow.xgboost
from mlflow.models import infer_signature



params = study.best_trial.params


with mlflow.start_run() as run:
    model = XGBClassifier(**params, n_jobs=-1)
    model.fit(X_over, y_over)

   
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)
    
    mlflow.log_params(params)
    mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred)})
    
    
    mlflow.xgboost.log_model(
        xgb_model = model,
        artifact_path="xgb-model",
        signature=signature,
        registered_model_name="xgb_tip_no_tip",
    )