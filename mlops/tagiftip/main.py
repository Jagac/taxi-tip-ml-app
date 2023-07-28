import warnings

import pandas as pd
from config import config

from tagiftip import data, train, utils

warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_parquet(f"{config.DATA_DIR}/yellow_tripdata_2023-04.parquet")
    df = data.preprocess(df)
    
    return df



if __name__ == "__main__":
    df = load_data()
    test = train.ModelTrainer(df)
    test.auto_train("test", "testauto")