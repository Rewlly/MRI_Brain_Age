import numpy
from numpy import ravel
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import BAE.io
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import seaborn as sns


def analysis(test, predict):
    y_grid = np.arange(min(test), max(test), 0.01)  # this step required because data is feature scaled.
    y_grid = y_grid.reshape((len(y_grid), 1))
    plt.scatter(y_test, y_pr_SVR, color='red')
    linear_regression_plot = LinearRegression()
    linear_regression_plot.fit(test, predict)
    plt.plot(y_grid, linear_regression_plot.predict(y_grid), color='blue')
    plt.plot(y_grid, y_grid, color='black')
    plt.title('Comparison of real age and estimation')
    plt.xlabel('Age')
    plt.ylabel('Age Estimation（Brain Age）')
    plt.annotate('MAE = {}'.format(mean_absolute_error(test, predict)), (1, 1))
    plt.annotate('r^2 = {}'.format(r2_score(test, predict)), (1, 1))
    plt.show()
    # sns.lmplot(test, predict)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The MAE
    print("Mean absolute error: %.2f" % mean_absolute_error(test, predict))
    # The median absolute error
    print("Median absolute error: %.2f" % median_absolute_error(test, predict))
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(test, predict))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(test, predict))


X_train, X_test, y_train, y_test = BAE.io.read_source_relative_site()

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.fit_transform(X_test)
y_train_sc = sc_y.fit_transform(y_train)
y_test_sc = sc_y.fit_transform(y_test)

scores = cross_val_score(XGBRegressor(), X_train, y_train, scoring='neg_mean_squared_error')
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, min_child_weight=5, max_depth=4)
xgb.fit(X_train, y_train)
print("Validation:", xgb.score(X_test, y_test))

SVR_regressor = SVR(kernel='rbf')
SVR_regressor.fit(X_train_sc, ravel(y_train_sc))

# Build a linear regression model for prediction
model = LinearRegression()
# x is a set of independent variables, y is the dependent variable
model.fit(X_train, y_train)
# The prediction of y based on x
y_pr_li = model.predict(X_test)

y_pr_xgb = xgb.predict(X_test)

y_pr_SVR_sc = SVR_regressor.predict(X_test_sc)
y_pr_SVR = sc_y.inverse_transform(y_pr_SVR_sc[:, numpy.newaxis])

analysis(y_test, y_pr_li)
analysis(y_test, y_pr_SVR)
analysis(y_test, y_pr_xgb)
