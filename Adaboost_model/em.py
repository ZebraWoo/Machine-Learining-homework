import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 高斯分布参数
mu = [-5, 5, 0]          # 均值
sigma_squared = [1.2, 1.8, 1.6]  # 方差
weights = [0.33, 0.33, 0.34]      # 每个组件的权重

# 生成数据
n_samples = 300
data = []
for i in range(n_samples):
    component = np.random.choice([0, 1, 2], p=weights)
    sample = np.random.normal(mu[component], np.sqrt(sigma_squared[component]))
    data.append(sample)

data = np.array(data)

# 绘制初始数据点分布
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Initial Data Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid()
plt.show()

# 高斯概率密度函数
def gaussian(x, mu, sigma_squared):
    return (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-0.5 * ((x - mu) ** 2 / sigma_squared))

def em_algorithm(data, n_components, n_iterations):
    # 初始化参数
    mu = np.random.choice(data, n_components)
    sigma_squared = np.random.rand(n_components) + 1  # 确保方差为正
    weights = np.ones(n_components) / n_components

    for iteration in range(n_iterations):
        # E步：计算后验概率
        responsibilities = np.zeros((data.shape[0], n_components))
        for k in range(n_components):
            responsibilities[:, k] = weights[k] * gaussian(data, mu[k], sigma_squared[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M步：更新参数
        N_k = responsibilities.sum(axis=0)
        for k in range(n_components):
            mu[k] = (responsibilities[:, k] * data).sum() / N_k[k]
            sigma_squared[k] = ((responsibilities[:, k] * (data - mu[k]) ** 2).sum() / N_k[k])
            weights[k] = N_k[k] / data.shape[0]

        # 打印当前参数
        print(f"Iteration {iteration + 1}:")
        for k in range(n_components):
            print(f"  Component {k + 1}: mu={mu[k]:.3f}, sigma^2={sigma_squared[k]:.3f}, weight={weights[k]:.3f}")

        # 绘制当前模型的分布
        x = np.linspace(-10, 10, 1000)
        pdf = sum(weights[k] * gaussian(x, mu[k], sigma_squared[k]) for k in range(n_components))
        plt.plot(x, pdf, label=f'Iteration {iteration + 1}')

    return mu, sigma_squared, weights

# 运行EM算法
n_components = 3
n_iterations = 10
em_algorithm(data, n_components, n_iterations)

# 最后绘制所有迭代的结果
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
plt.title('GMM Convergence Over Iterations')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()
