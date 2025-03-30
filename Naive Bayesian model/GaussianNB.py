from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 加载数据
import pandas as pd
data = pd.read_excel('Dry_Bean_Dataset.xlsx')

# 分离特征和标签
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练朴素贝叶斯模型
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 预测
y_pred_nb = nb_model.predict(X_test)

# 评估模型
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report for Naive Bayes:\n", classification_report(y_test, y_pred_nb))
'''
朴素贝叶斯：在高维数据上计算效率高，但由于特征独立性假设，精度可能不如其他模型。
线性分类器：适合线性可分数据，简单且高效，但在数据非线性时效果较差。
决策树：易解释，但容易过拟合，适合处理复杂数据结构。
SVM：能很好地处理复杂的决策边界，适合高维数据，但计算复杂度高。
神经网络：适合复杂模式学习，但需要大量计算资源和较长训练时间。
'''