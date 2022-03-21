import pandas as pd
import numpy as np
import sklearn.feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from BAE.io import read_source


def main():
    X_train, X_test, y_train, y_test = read_source()

    # apply SelectKBest class to extract top 10 best features
    # best_features = SelectKBest(score_func=f_regression(X_train, y_train), k=10)
    sklearn.feature_selection.f_regression(X_train, y_train)
    X_train_select = SelectKBest(score_func=f_regression, k=20).fit(X_train, y_train)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X_train.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([df_columns, df_scores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


if __name__ == '__main__':
    main()
