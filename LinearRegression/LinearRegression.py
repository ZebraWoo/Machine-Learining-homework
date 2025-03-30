import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# 加载训练数据和测试数据
train_data = pd.read_csv('prostate_train.txt', sep='\t')
test_data = pd.read_csv('prostate_test.txt', sep='\t')

# 选取前四列作为特征，最后一列作为目标
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 输出模型系数
print("模型系数: ", model.coef_)
print("截距: ", model.intercept_)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
rss = mse * len(y_test)  # 残差平方和
print("RSS: ", rss)

# 二次项（包括交叉项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 训练新的线性回归模型
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# 预测并计算 RSS
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rss_poly = mse_poly * len(y_test)
print("考虑交叉项的 RSS: ", rss_poly)

