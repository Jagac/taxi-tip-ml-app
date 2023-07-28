import numpy as np
import pandas as pd
from config.config import logger
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def preprocess(df):
    df["tipped"] = (df["tip_amount"] > 0).astype("int")
    df["trip_time"] = (df.tpep_pickup_datetime - df.tpep_dropoff_datetime).astype(
        "timedelta64[s]") / np.timedelta64(1, "s")
    df = df.drop(
        ["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount"], axis=1
    )
    one_hot_enc = OneHotEncoder()
    arr = one_hot_enc.fit_transform(df[["store_and_fwd_flag"]])
    store_and_fwd_flag = pd.DataFrame(arr, columns=["store_and_fwd_flag_ohe"])
    df_merge = pd.merge(df, store_and_fwd_flag, left_index=True, right_index=True)
    df_merge = df_merge.reset_index(drop=True)
    df_merge = df_merge.drop("store_and_fwd_flag", axis=1)


    logger.info("Preprocessing complete!")
    return df_merge


def get_data_splits(X, y, train_size=0.7):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y
    )

    logger.info("Data split complete!")
    return X_train, X_test, y_train, y_test


def oversample_data(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    logger.info("Balancing complete!")
    return X_over, y_over
