import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data.csv")

columns = ['Survived', 'Pclass', 'Sex', 'Age']

Y = pd.DataFrame()
Y["survived"] = df["Survived"]

X = pd.DataFrame()
X["pclass"] = df["Pclass"]
X["sex"] = df["Sex"]
X["age"] = df["Age"]

X["age"] = X["age"].fillna(X["age"].median())


sex_cols = ["male", "female"]
for i in range(len(X)):
    X["sex"][i] = sex_cols.index(X["sex"][i])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model_lr = LogisticRegression()
model_knn = KNeighborsClassifier()

model_lr.fit(X_train, Y_train)
model_knn.fit(X_train, Y_train)

Y_pred_lr = model_lr.predict(X_test)
Y_pred_knn = model_knn.predict(X_test)

matrix_lr = confusion_matrix(Y_test, Y_pred_lr)
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)

print(matrix_lr)
print(matrix_knn)