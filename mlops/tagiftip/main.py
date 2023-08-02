import pandas as pd

from config import config
from tagiftip import data, registry, train, utils


def load_data():
    df = pd.read_parquet(f"{config.DATA_DIR}/yellow_tripdata_2023-04.parquet")
    df = data.preprocess(df)

    return df


def auto_train(df):
    model = train.ModelTrainer(df)
    model.auto_train(run_name="xgb_auto_tune", experiment_name="xgb_auto_tune")


def manual_train(
    df,
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
):
    model = train.ModelTrainer(df)
    model.manual_train(
        run_name=run_name,
        experiment_name=experiment_name,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )


def load_model(name, version):
    reg = registry.RegistryManipulator(name)
    model = reg.fetch_model_from_registry(model_version=version)
    reg.get_model_info()

    return model


def predict(model, data):
    result = model.predict(data)

    return result


def transition_stage(name):
    model = registry.RegistryManipulator(name)
    model.transition_model_stage(1, "Production")


if __name__ == "__main__":
    df = load_data()
    df = df.drop("tipped", axis=1)
    df = df.head(1)
    # auto_train(df)
    # transition_stage("xgb_tip_no_tip")

    model = load_model("xgb_tip_no_tip", 1)
    result = predict(model=model, data=df)
    print(result)
