import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.svm import SVR

data1 = pd.read_csv(r"\AD\sMCI_1styear.csv")
data2 = pd.read_csv(r"\AD\sMCI_2ndyear.csv")
data3 = pd.read_csv(r"\AD\sMCI_3rdyear.csv")
data4 = pd.read_csv(r"\AD\pMCI_2yearsbefore.csv")
data5 = pd.read_csv(r"C:\Users\Han\PycharmProjects\MRI_Brain_Age\AD\pMCI_1yearsbefore.csv")
data6 = pd.read_csv(r"C:\Users\Han\PycharmProjects\MRI_Brain_Age\AD\pMCI_0yearsbefore.csv")

data_ls = [data1, data2, data3, data4, data5, data6]

for data in data_ls:
    X_test = data.drop(columns=['name', 'age', 'sex', 'WM', 'GM', 'CSF'])
    y_test = data['age']
    #     y_pred = np.zeros(len(y_test))
    model = joblib.load(r"SVR-3")
    y_pred = model.predict(X_test)
    #     y_pred = y_pred + y_pre
    data['bias'] = y_pred - y_test

sMCI1 = data_ls[0]['bias']
sMCI2 = data_ls[1]['bias']
sMCI3 = data_ls[2]['bias']
pMCI1 = data_ls[3]['bias']
pMCI2 = data_ls[4]['bias']
pMCI3 = data_ls[5]['bias']

data = pd.DataFrame()
data['sMCI1'] = sMCI1
data['sMCI2'] = sMCI2
data['sMCI3'] = sMCI3
data['pMCI1'] = pMCI1
data['pMCI2'] = pMCI2
data['pMCI3'] = pMCI3

data.to_csv(r"./PAD.csv", index=False)
