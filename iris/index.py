import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("IRIS.csv")

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

Y = pd.DataFrame()
Y["species"] = df["species"]
X = df.drop("species", axis=1)

Y_cols = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i in range(len(Y)):
    Y["species"][i] = Y_cols.index(Y["species"][i]) + 1

Y["species"] = Y["species"].astype("int")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.14, random_state=0)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_dt = DecisionTreeClassifier(random_state=0)
model_rf = RandomForestClassifier(n_estimators=141, max_depth=4)

model_knn.fit(X_train, Y_train.values.ravel())
model_dt.fit(X_train, Y_train)
model_rf.fit(X_train, Y_train)

Y_pred_knn = model_knn.predict(X_test)
Y_pred_dt = model_dt.predict(X_test)
Y_pred_rf = model_rf.predict(X_test)

matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
matrix_dt = confusion_matrix(Y_test, Y_pred_dt)
matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
print(matrix_knn)
print(matrix_dt)
print(matrix_rf)