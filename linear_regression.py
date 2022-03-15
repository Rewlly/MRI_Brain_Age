# This is a prediction of brain age based on linear regression using the scikit-learn module.
# Date: 3/14/2022
# Author: Han

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # Source is data from 8716 subjects, with MRI data and age, gender
    source = pd.read_csv(
        r'volume_sum_icv_site.csv')
    y = source.iloc[:, 102]  # y is subjects' age
    x = source.iloc[:, 1:100]  # x consists of 95 labels and 5 sum features

    # Split train and test data set
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Build a linear regression model for prediction
    model = LinearRegression()
    # x is a set of independent variables, y is the dependent variable
    model.fit(x_train, y_train)
    # The prediction of y based on x
    y_predict = model.predict(x_test)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The MAE
    print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_predict))
    # The median absolute error
    print("Median absolute error: %.2f" % median_absolute_error(y_test, y_predict))
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_predict))
