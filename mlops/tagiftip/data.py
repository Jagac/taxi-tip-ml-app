import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config.config import logger


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data so it can be used by the model

    Args:
        df (pd.DataFrame): original dataset.

    Returns:
        pd.DataFrame: transformed dataset.
    """
    df["tipped"] = (df["tip_amount"] > 0).astype("int")
    df["trip_time"] = (df.tpep_pickup_datetime - df.tpep_dropoff_datetime).astype(
        "timedelta64[s]"
    ) / np.timedelta64(1, "s")

    df = df.drop(
        ["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount"], axis=1
    )

    ohe_df = pd.get_dummies(df.store_and_fwd_flag, prefix="ohe", drop_first=True)
    df_merge = pd.concat([df, ohe_df], axis=1)

    df_merge = df_merge.drop("store_and_fwd_flag", axis=1)

    logger.info("Preprocessing complete!")
    return df_merge


def get_data_splits(X: pd.DataFrame, y: pd.DataFrame, train_size: float) -> np.array:
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y
    )

    logger.info("Data split complete!")
    return X_train, X_test, y_train, y_test


def oversample_data(X_train: np.array, y_train: np.array) -> np.array:
    # performs oversampling on undersampled data
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    logger.info("Balancing complete!")
    return X_over, y_over
