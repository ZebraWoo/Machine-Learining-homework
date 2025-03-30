import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 设置随机种子以便复现
np.random.seed(42)

# 生成仿真数据函数
def generate_data(n_samples, sigma):
    x = np.random.normal(0, 1, n_samples)
    epsilon = np.random.normal(0, sigma, n_samples)
    y = 3 * x + 6 + epsilon  # θ_1 = 3, θ_0 = 6
    return x, y



# 生成训练数据
x_train, y_train = generate_data(10, sigma=0.5)  # 生成10个训练数据，σ = 0.5
x_train = x_train[:, np.newaxis]  # 转换为列向量

# # 生成100个训练样本
# x_train_100, y_train_100 = generate_data(100, sigma=0.5)
# x_train_100 = x_train_100[:, np.newaxis]

# 拟合线性回归模型
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_train)

# 二次回归模型
poly_2 = PolynomialFeatures(degree=2)
x_train_poly_2 = poly_2.fit_transform(x_train)
quad_model = LinearRegression()
quad_model.fit(x_train_poly_2, y_train)
y_pred_quad = quad_model.predict(x_train_poly_2)

# 三次回归模型
poly_3 = PolynomialFeatures(degree=3)
x_train_poly_3 = poly_3.fit_transform(x_train)
cubic_model = LinearRegression()
cubic_model.fit(x_train_poly_3, y_train)
y_pred_cubic = cubic_model.predict(x_train_poly_3)

rss_linear = np.sum((y_train - y_pred_linear) ** 2)
rss_quad = np.sum((y_train - y_pred_quad) ** 2)
rss_cubic = np.sum((y_train - y_pred_cubic) ** 2)

print(f"线性模型 RSS: {rss_linear}")
print(f"二次模型 RSS: {rss_quad}")
print(f"三次模型 RSS: {rss_cubic}")

# 绘制散点图和回归曲线
plt.scatter(x_train, y_train, color='black', label='数据点')

# 生成用于绘制平滑曲线的 x 值
x_range = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)

# 线性模型的回归曲线
plt.plot(x_range, linear_model.predict(x_range), color='blue', label='线性回归')

# 二次模型的回归曲线
x_range_poly_2 = poly_2.transform(x_range)
plt.plot(x_range, quad_model.predict(x_range_poly_2), color='green', label='二次回归')

# 三次模型的回归曲线
x_range_poly_3 = poly_3.transform(x_range)
plt.plot(x_range, cubic_model.predict(x_range_poly_3), color='red', label='三次回归')

plt.xlabel('x')
plt.ylabel('y')
plt.title('训练数据和回归曲线')
plt.legend()
plt.show()
