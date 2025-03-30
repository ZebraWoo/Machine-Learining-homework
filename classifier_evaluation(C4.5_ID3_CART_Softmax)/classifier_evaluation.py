import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# 加载数据
data = pd.read_csv("drug200.csv")

# 数据预处理：将类别型特征转换为数值型
label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 分离特征和目标列
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化结果列表
results = []

# 定义分类器列表
classifiers = {
    "ID3": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "C4.5": DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42),
    "CART": DecisionTreeClassifier(criterion='gini', random_state=42),
    "Softmax Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=800, random_state=42)
}

# 训练和评估每个分类器
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} Accuracy:", accuracy)
    results.append({'Model': name, 'Accuracy': accuracy, **report['weighted avg']})

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 保存结果到 CSV 文件
results_df.to_csv('classifier_evaluation_results.csv', index=False)
print("Evaluation results saved to 'classifier_evaluation_results.csv'.")

# 可视化准确率对比
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Accuracy Comparison of Different Classifiers')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.show()

# 可视化各模型的精确率、召回率和F1-score对比
metrics = ['precision', 'recall', 'f1-score']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(f'{metric.capitalize()} Comparison of Different Classifiers')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.show()
