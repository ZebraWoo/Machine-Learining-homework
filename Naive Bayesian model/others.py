from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 定义模型字典
models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'decision_tree': DecisionTreeClassifier(),
    'mlp': MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500),
    'svm_linear': SVC(kernel='linear', probability=True),
    'svm_rbf': SVC(kernel='rbf', probability=True)
}


# 定义模型选择和评估函数
def evaluate_model(model_name, X_train, y_train, X_test, y_test):
    model = models.get(model_name)

    if model is None:
        print("无效的模型选择，请从以下模型中选择：", list(models.keys()))
        return

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)


    print(f"模型：{model_name}")
    print("分类报告：\n", classification_report(y_test, y_pred))
    print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))

    # 计算AUC-ROC分数
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"AUC-ROC 分数: {roc_auc}")
    else:
        print("该模型不支持AUC-ROC分数计算。")





url = "Dry_Bean_Dataset.xlsx"
data = pd.read_excel(url,'Dry_Beans_Dataset')

X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签（豆类的种类）



print(data.head())
le = LabelEncoder()
y = le.fit_transform(y)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 用户选择模型
model_name = input(f"请选择模型 {list(models.keys())}: ")

# 调用模型评估函数
evaluate_model(model_name, X_train, y_train, X_test, y_test)