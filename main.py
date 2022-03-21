# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from __future__ import annotations

import pandas
from sklearn.model_selection import train_test_split


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_source_MCI(csv_path: str | None = "volume_sum_icv_site.csv"):
    """

    :param csv_path: str
    :return: Train and test dataset of independent variables X and dependent variable y
    """

    MRI_source_df = pandas.read_csv(csv_path)
    X_df = MRI_source_df.iloc[:, 2:-2]
    del X_df['age']
    y_df = MRI_source_df.iloc[:, -1]
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
