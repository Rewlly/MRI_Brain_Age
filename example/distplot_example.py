import seaborn as sns
import matplotlib.pyplot as plt

# 画密度图，也可以用更底层的plt.hist(trainDF['SalePrice'])来做
sns.distplot(trainDF['SalePrice'])

# 因为seaborn是matplotlib的高级库，所以可以用plt.show()来调动绘图
plt.show()