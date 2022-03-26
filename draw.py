######################################PAPER
from __future__ import annotations

from sklearn.preprocessing import StandardScaler
import numpy
import numpy as np
import pandas as pd
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from partd import pandas
from scipy import stats
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import norm
from scipy import special
import joblib
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle
import math
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def remove_fliers_with_boxplot(data, col):
    p = data[col].boxplot(return_type='dict')
    # 获取异常值
    for index, value in enumerate(data[col].columns):
        fliers_value_list = p['fliers'][index].get_ydata()
        # 删除异常值
        for flier in fliers_value_list:
            data = data[data.loc[:, value] != flier]

    return data


def ZscoreNormalization(x, mean, std):
    """Z-score normaliaztion"""
    x = (x - mean) / std
    return x


def MinMaxNormalization(x, Max, Min):
    """Z-score normaliaztion"""
    x = (x - Min) / (Max - Min)
    return x


def plot_regression_results(ax, y_true, y_pred, scores):  # ax子图像
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],  # y=x线，范围：实际年龄
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)
    #####################坐标轴设置#################
    ax.spines['top'].set_visible(False)  # spine：脊。去掉top和right边框(隐藏坐标轴)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()  # 底边框作为x(自变量)轴，左边框作为y(因变量)轴(x.y轴绑定)
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))  # 将x,y轴绑定到特定位置(交点为(10,10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])  # 设置坐标轴刻度(范围，真实年龄的最大、最小值)
    ax.set_ylim([y_true.min(), y_true.max()])
    # ax.set_xlabel('Measured')
    # #设置x、y轴名称 ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,  # 画矩形图，画图起点(0,0),矩形宽度0？？？？？？？？？？？、矩形高度0？？？？？？？？？？？？？？
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left', fontsize=16)


def get_p(groupA, groupB):
    all_in = groupA.append(groupB)
    all_labels = [0 for i in range(len(groupA))] + [1 for i in range(len(groupB))]
    mc = MultiComparison(all_in, all_labels)
    result = mc.tukeyhsd()

    return result.pvalues[0]


def get_spe(y_true, y_pred):
    # true positive
    TP = np.sum(np.multiply(y_true, y_pred))  ##########数组对应位置元素相乘 1→1
    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  ############逐元素计算逻辑与 0→1
    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  ###########1→0
    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  ##########0→0

    spe = TN / (TN + FP)
    return spe


def read_source_relative_site(csv_path: str | None = "volume_sum_icv_site.csv"):
    """

    :param csv_path: str
    :return: Train and test dataset of independent variables X and dependent variable y, where X is labels' relative
    value to ICV.
    """

    MRI_source_df = pandas.read_csv(csv_path)
    MRI_source_df = MRI_source_df[MRI_source_df['sites'].isin([1])]
    X_df = MRI_source_df.iloc[:, 1:-8]
    icv_df = MRI_source_df.iloc[:, -4]
    y_df = MRI_source_df.iloc[:, -2]
    X = X_df
    y = y_df
    icv = icv_df
    icv = icv[:, np.newaxis]
    y = y[:, np.newaxis]
    X = X / icv
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X, X, y, y


# 这一部分读入你的ADNI(sites=1)相对体积数据，并进行train_test_split
train = pd.read_csv('volume_sum_icv_site.csv')
train = train[train['sites'].isin([1])]
X_train = train.drop(columns=['name', 'age', 'sex', 'sites', 'GM', 'WM', 'CSF', 'ICV', 'sum_vol'])
X_train = (X_train / train['ICV'][:, np.newaxis]).iloc[0:1155, :]
y_train = (train['age'])
y_train = y_train.iloc[0:1155]
X_test = train.drop(columns=['name', 'age', 'sex', 'sites', 'GM', 'WM', 'CSF', 'ICV', 'sum_vol'])
X_test = (X_test / train['ICV'][:, np.newaxis]).iloc[1155:-1, :]
y_test = (train['age'])
y_test = y_test.iloc[1155:-1]
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.fit_transform(X_test)
y_train_axis = y_train[:, np.newaxis]
y_test_axis = y_test[:, np.newaxis]
y_train_np = pd.Series.to_numpy(y_train)
y_test_np = pd.Series.to_numpy(y_test)
y_train_re = np.reshape(y_train_np, (-1, 1))
y_test_re = np.reshape(y_test_np, (-1, 1))
y_train_sc = sc_y.fit_transform(y_train_re)
y_test_sc = sc_y.fit_transform(y_test_re)


###########训练模型
 #model = xgb.XGBRegressor()
 #model = LinearRegression()
model = SVR()
# model = joblib.load(r'SVR')
# model = joblib.load(r'linear')
# model = joblib.load(r'xgb')

model.fit(X_train_sc, y_train_sc)
# y_pred_sc = model.predict(X_test_sc)
# y_pred_sc_axis = y_pred_sc[:, np.newaxis]
# y_pred = sc_y.inverse_transform(y_pred_sc)
y_pred_sc = model.predict(X_test_sc)
y_pred_2d = np.reshape(y_pred_sc, (-1, 1))
y_pred = sc_y.inverse_transform(y_pred_2d)

# print("Pearson:", np.corrcoef(np.array([y_test, y_pred])))
# df = pd.DataFrame()
# df['y_true'] = y_test
# df['y_pred'] = y_pred
# df.to_csv(r"D:\rc\毕设\CSV\prediction.csv", index=False)
fig, axs = plt.subplots(1, 1, figsize=(5, 3.5), dpi=200)
axs = np.ravel(axs)
for ax in axs:
    score1 = r2_score(y_test, y_pred)
    score2 = mean_absolute_error(y_test, y_pred)
    plot_regression_results(ax, y_test, y_pred,
                            (r'$R^2={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(score1, score2))


# plt.title('模型测试结果', fontsize=16)


def Fun(x, a1, a2):  # 定义拟合函数形式
    return a1 * x + a2


x = pd.Series.to_numpy(y_test)  # 创建时间序列
a1, a2 = [1, 5]  # 原始数据的参数
y = y_pred
y = y.squeeze()
para, pcov = curve_fit(Fun, x, y)
y_fitted = Fun(x, para[0], para[1])  # 画出拟合后的曲线
plt.plot(x, y_fitted, '-g', label='Fitted curve')
print(para)
# plt.tight_layout()                                                                                          
# plt.subplots_adjust(top=0.9)                                                                                
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Chronological age', fontsize=8)
plt.ylabel('Predicted age', fontsize=8)
# plt.savefig(r"D:\rc\paper\fig\model.png", dpi=400)
plt.show()
