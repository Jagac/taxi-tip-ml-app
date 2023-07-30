import mlflow
import pandas as pd
from config.config import MODEL_REGISTRY
from mlflow import MlflowClient

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
                print(info)

    def fetch_model_from_registry(self, model_version: int) -> None:
        # mlflow.artifacts.download_artifacts("file:///home/jagac/projects/taxi-tip-mlapp/Research/mlruns/995761978312635047/22bae30ec72e43299490cfa64f20ec20/artifacts/xgb-model")
        self.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_name}/{model_version}"
        )

    def make_prediction(self, df: pd.DataFrame) -> None:
        result = self.model.predict(df)
        
        return result
