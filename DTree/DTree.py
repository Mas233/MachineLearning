import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt

all_data=pd.read_csv('data/titanic3.csv')

X=all_data[["pclass", "sex", "age", "sibsp", "parch"]]
Y=all_data["survived"]

X["age"].fillna(X["age"].median(),inplace=True)
X=pd.get_dummies(X,columns=["sex"],drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

classifier=DecisionTreeClassifier(random_state=42)

classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

accuracy=accuracy_score(Y_test,y_pred)
confusion = confusion_matrix(Y_test, y_pred)
report = classification_report(Y_test, y_pred)

print("Accuracy: ",accuracy)
print("Confusion Matrix: \n",confusion)
print("Classification Report: \n",report)

plt.figure(figsize=(15,8))
plot_tree(classifier,feature_names=X.columns,class_names=["Not Survived","Survived"],filled=True)
plt.show()