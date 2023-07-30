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


def manual_train(df):
    model = train.ModelTrainer(df)
    model.manual_train(run_name="xgb_manual", experiment_name="xgb_manual")


def load_model(name, version, df):
    model = registry.RegistryManipulator(name)
    model.fetch_model_from_registry(model_version=version)
    model.get_model_info()
    result = model.make_prediction(df)
    
    print(result)
   

def transition_stage(name):
    model = registry.RegistryManipulator(name)
    model.transition_model_stage(1, "Production")


    

if __name__ == "__main__":
    df = load_data()
    #auto_train(df)
    #transition_stage("xgb_tip_no_tip")
    model = load_model("xgb_tip_no_tip", 1, df)
    
