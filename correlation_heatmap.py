import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("volume_sum_icv_site.csv")
# X = data.iloc[:, 0:20]  # independent columns
# y = data.iloc[:, -1]  # target column i.e price range

# get correlations of each features in dataset
correlation_mat = data.corr()
top_corr_features = correlation_mat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
