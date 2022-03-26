from __future__ import annotations
import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def read_source(csv_path: str | None = "volume_sum_icv_site.csv"):
    """

    :param csv_path: str
    :return: Train and test dataset of independent variables X and dependent variable y
    """

    MRI_source_df = pandas.read_csv(csv_path)
    X_df = MRI_source_df.iloc[:, 1:-1]
    del X_df['age']
    y_df = MRI_source_df.iloc[:, -2]
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def read_source_relative(csv_path: str | None = "volume_sum_icv_site.csv"):
    """

    :param csv_path: str
    :return: Train and test dataset of independent variables X and dependent variable y, where X is labels' relative
    value to ICV.
    """

    MRI_source_df = pandas.read_csv(csv_path)
    X_df = MRI_source_df.iloc[:, 1:-9]
    icv_df = MRI_source_df.iloc[:, -4]
    y_df = MRI_source_df.iloc[:, -2]
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    icv = icv_df.to_numpy()
    icv = icv[:, np.newaxis]
    y = y[:, np.newaxis]
    X = X / icv
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def read_source_relative_site(csv_path: str | None = "volume_sum_icv_site.csv"):
    """

    :param csv_path: str
    :return: Train and test dataset of independent variables X and dependent variable y, where X is labels' relative
    value to ICV.
    """

    MRI_source_df = pandas.read_csv(csv_path)
    MRI_source_df = MRI_source_df[MRI_source_df['sites'].isin([1])]
    X_df = MRI_source_df.iloc[:, 1:-9]
    icv_df = MRI_source_df.iloc[:, -4]
    y_df = MRI_source_df.iloc[:, -2]
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    icv = icv_df.to_numpy()
    icv = icv[:, np.newaxis]
    y = y[:, np.newaxis]
    X = X / icv
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
