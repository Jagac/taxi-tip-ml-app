import mlflow
import pandas as pd
from mlflow import MlflowClient

from config.config import MODEL_REGISTRY

mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))


class RegistryManipulator:
    def __init__(self, model_name: str, client=MlflowClient()) -> None:
        self.model_name = model_name
        self.client = client

    def register_model(self, run_id: str) -> None:
        result = mlflow.register_model(
            "file:///home/jagac/projects/taxi-tip-mlapp/Research/mlruns", f"{run_id}"
        )

    def transition_model_stage(self, version: int, stage: str) -> None:
        self.client.transition_model_version_stage(
            name=f"{self.model_name}", version=version, stage=f"{stage}"
        )

    def get_model_info(self: None) -> None:
        for mv in self.client.search_model_versions(f"name='{self.model_name}'"):
            info = dict(mv)
            if info["current_stage"] == "Production":
                print(f"[MODEL INFO] : {info}")

    def fetch_model_from_registry(self, model_version: int) -> None:
        mlflow.artifacts.download_artifacts(
            "file:///home/jagac/projects/taxi-tip-mlapp/Research/mlruns/654781923270945138/61516d4c3e6848068b6b1613b3f8b355/artifacts/xgb-model-2023-07-30 12:30:24.166188"
        )
        self.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_name}/{model_version}"
        )

        return self.model

    def make_prediction(self, df: pd.DataFrame) -> None:
        result = self.model.predict(df)

        return result
