import pandas as pd

# 加载数据集
data = pd.read_excel('Dry_Bean_Dataset.xlsx')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 划分数据集
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建深度不超过2的决策树分类器
dt_classifier = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_classifier.fit(X_train, y_train)

# 评估模型性能
dt_accuracy = dt_classifier.score(X_test, y_test)
print(f'Decision Tree Accuracy: {dt_accuracy}')

#Ada
from sklearn.ensemble import AdaBoostClassifier

# 创建Adaboost模型
ada_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42)
ada_classifier.fit(X_train, y_train)

# 评估模型性能
ada_accuracy = ada_classifier.score(X_test, y_test)
print(f'AdaBoost Accuracy: {ada_accuracy}')

#Bagging
from sklearn.ensemble import BaggingClassifier

# 创建Bagging模型
bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42)
bagging_classifier.fit(X_train, y_train)

# 评估模型性能
bagging_accuracy = bagging_classifier.score(X_test, y_test)
print(f'Bagging Accuracy: {bagging_accuracy}')

#RF
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林模型
rf_classifier = RandomForestClassifier(max_depth=2, random_state=42)
rf_classifier.fit(X_train, y_train)

# 评估模型性能
rf_accuracy = rf_classifier.score(X_test, y_test)
print(f'Random Forest Accuracy: {rf_accuracy}')

print(f'准确率比较:')
print(f'决策树: {dt_accuracy}')
print(f'Adaboost: {ada_accuracy}')
print(f'Bagging: {bagging_accuracy}')
print(f'随机森林: {rf_accuracy}')
