import datetime

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import accuracy_score

from config.config import logger
from tagiftip import data, utils


class ModelTrainer:
    def __init__(self, df: pd.DataFrame) -> None:
        utils.set_seeds()
        self.y = df.tipped
        self.X = df.drop("tipped", axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = data.get_data_splits(
            self.X, self.y, train_size=0.7
        )
        self.X_over, self.y_over = data.oversample_data(self.X_train, self.y_train)

    def objective(self, trial: optuna.trial) -> float:
        """Hyperparameter tuning experiment.

        Args:
            trial (optuna.trial): instance represents a process of evaluating an objective function.

        Returns:
            int: accuracy on the test set
        """
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
        }

        optuna_model = xgb.XGBClassifier(**params)
        optuna_model.fit(self.X_over, self.y_over)

        y_pred = optuna_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        return accuracy

    def optimize_params(
        self, study_name: str, num_trials: str, objective: callable
    ) -> dict:
        """Apply the objective function and start training

        Args:
            study_name (str): name of the study.
            num_trials (str): number of hyperparameter searches ie how many times to train.
            objective (function): objective function

        Returns:
            dict: best parameters.
        """

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(
            study_name=study_name, direction="maximize", pruner=pruner
        )
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="accuracy"
        )

        study.optimize(
            objective, n_trials=num_trials, callbacks=[mlflow_callback], n_jobs=-1
        )

        return study.best_trial.params

    def auto_train(self, run_name: str, experiment_name: str) -> None:
        """Performs automatic tuning and logs the model with best hyperparameters

        Args:
            run_name (str): name of the run with the best params
            experiment_name (str): name of the experiment for model with best params
        """

        logger.info("Optimization started!")
        params = self.optimize_params(
            "auto_train", num_trials=2, objective=self.objective
        )
        mlflow.set_experiment(experiment_name=experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            model = xgb.XGBClassifier(**params, n_jobs=-1)
            model.fit(self.X_over, self.y_over)

            y_pred = model.predict(self.X_test)
            signature = mlflow.models.infer_signature(self.X_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metrics({"accuracy": accuracy_score(self.y_test, y_pred)})

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=f"xgb-model-{datetime.datetime.now()}",
                signature=signature,
                registered_model_name="xgb_tip_no_tip",
            )

        logger.info("Optimization complete!")

    def manual_train(
        self,
        run_name: str,
        experiment_name: str,
        max_depth: int,
        learning_rate: float,
        n_estimators: int,
        min_child_weight: int,
        gamma: float,
        subsample: float,
        colsample_bytree: float,
        reg_alpha,
        reg_lambda: float,
    ) -> None:
        """Performs manual training, requires specific hyperparameters
        Args:
            https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        """

        logger.info("Manual training started!")

        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }

        mlflow.set_experiment(experiment_name=experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            model = xgb.XGBClassifier(**params, n_jobs=-1)
            model.fit(self.X_over, self.y_over)

            y_pred = model.predict(self.X_test)
            signature = mlflow.models.infer_signature(self.X_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metrics({"accuracy": accuracy_score(self.y_test, y_pred)})

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=f"xgb-model-{datetime.datetime.now()}",
                signature=signature,
            )

        logger.info("Manual training complete!")

        return model
