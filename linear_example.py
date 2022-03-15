import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ————————————————
# 版权声明：本文为CSDN博主「心态与做事习惯决定人生高度」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/robert_chen1988/article/details/103551261
if __name__ == '__main__':
    source = pd.read_csv(
        r'volume_sum_icv_site.csv')  # 读取 excel 数据，引号里面是 excel 文件的位置
    y = source.iloc[:, 102]  # 因变量为第 2 列数据
    x = source.iloc[:, 1:95]  # 自变量为第 2~96 列数据
    plt.scatter(x, y, label='实际值')  # 散点图

    # 将 x，y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
    x1 = x[:, np.newaxis]
    y1 = y[:, np.newaxis]

    model = LinearRegression()  # 构建线性模型
    model.fit(x1, y1)  # 自变量在前，因变量在后
    predicts = model.predict(x1)  # 预测值
    R2 = model.score(x1, y1)  # 拟合程度 R2
    print('R2 = %.2f' % R2)  # 输出 R2
    coef = model.coef_  # 斜率
    intercept = model.intercept_  # 截距
    print(model.coef_, model.intercept_)  # 输出斜率和截距

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    y = source.iloc[:, 1]  # 因变量为第 2 列数据
    x = source.iloc[:, 2]  # 自变量为第 3 列数据

    plt.plot(x, predicts, color='red', label='预测值')
    plt.legend()  # 显示图例，即每条线对应 label 中的内容
    plt.show()  # 显示图形
