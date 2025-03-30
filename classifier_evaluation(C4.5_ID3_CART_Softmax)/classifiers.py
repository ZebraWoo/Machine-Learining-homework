import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("drug200.csv")

label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ID3 决策树
id3_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_tree.fit(X_train, y_train)

# 可视化 
plt.figure(figsize=(16, 10))
plot_tree(id3_tree, 
          feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
          class_names=label_encoders['Drug'].classes_,
          filled=True)
plt.title('ID3 Decision Tree (Information Gain)')
plt.show()

# 评估 
y_pred_id3 = id3_tree.predict(X_test)
print("ID3 Decision Tree Classification Report:\n", classification_report(y_test, y_pred_id3))
print("ID3 Accuracy:", accuracy_score(y_test, y_pred_id3))

# C4.5 决策树
c45_tree = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
c45_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 10))
plot_tree(c45_tree, 
          feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
          class_names=label_encoders['Drug'].classes_,
          filled=True)
plt.title('C4.5 Decision Tree (Information Gain Ratio)')
plt.show()

y_pred_c45 = c45_tree.predict(X_test)
print("C4.5 Decision Tree Classification Report:\n", classification_report(y_test, y_pred_c45))
print("C4.5 Accuracy:", accuracy_score(y_test, y_pred_c45))

# CART 决策树
cart_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 10))
plot_tree(cart_tree, 
          feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
          class_names=label_encoders['Drug'].classes_,
          filled=True)
plt.title('CART Decision Tree (Gini Index)')
plt.show()

y_pred_cart = cart_tree.predict(X_test)
print("CART Decision Tree Classification Report:\n", classification_report(y_test, y_pred_cart))
print("CART Accuracy:", accuracy_score(y_test, y_pred_cart))

# Softmax 
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
softmax_model.fit(X_train, y_train)

y_pred_softmax = softmax_model.predict(X_test)
print("Softmax Regression Classification Report:\n", classification_report(y_test, y_pred_softmax))
print("Softmax Regression Accuracy:", accuracy_score(y_test, y_pred_softmax))

conf_matrix = confusion_matrix(y_test, y_pred_softmax)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Softmax Regression Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()