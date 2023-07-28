from mlflow import MlflowClient
import mlflow
from config.config import MODEL_REGISTRY

mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

def register_model(run_id):
    result = mlflow.register_model(
    "file:///home/jagac/projects/taxi-tip-mlapp/Research/mlruns", f"{run_id}"
    )


def transition_model_stage(run_id, version, stage):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=f"{run_id}", version=version, stage=f"{stage}"
    )

    

def get_model_info(model_name):
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        test = dict(mv)
        if test['current_stage'] == "Staging":
            print(test['run_id'])

  

