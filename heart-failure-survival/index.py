import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv("data.csv")
df.drop(["platelets", "serum_sodium", "time"], axis=1, inplace=True)

Y = pd.DataFrame()
Y["died"] = df["DEATH_EVENT"]

X = pd.DataFrame()
X["age"] = df["age"]
X["anaemia"] = df["anaemia"]
X["creatinine_phosphokinase"] = df["creatinine_phosphokinase"]
X["diabetes"] = df["diabetes"]
X["ejection_fraction"] = df["ejection_fraction"]
X["high_blood_pressure"] = df["high_blood_pressure"]
X["serum_creatinine"] = df["serum_creatinine"]
X["sex"] = df["sex"]
X["smoking"] = df["smoking"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.14)

model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()
model_knn = KNeighborsClassifier()
model_rf = RandomForestClassifier(n_estimators=141, max_depth=8)

ensemble_model = VotingClassifier(
    estimators=[
        ("lr", model_lr),
        ("dt", model_dt),
        ("knn", model_knn),
        ("rf", model_rf),
    ],
    voting="hard",
)

model_lr.fit(X_train, Y_train)
model_dt.fit(X_train, Y_train)
model_knn.fit(X_train, Y_train)
model_rf.fit(X_train, Y_train)
ensemble_model.fit(X_train, Y_train)

Y_pred_lr = model_lr.predict(X_test)
Y_pred_dt = model_dt.predict(X_test)
Y_pred_knn = model_knn.predict(X_test)
Y_pred_rf = model_rf.predict(X_test)
Y_pred_ensemble = ensemble_model.predict(X_test)

matrix_lr = confusion_matrix(Y_test, Y_pred_lr)
matrix_dt = confusion_matrix(Y_test, Y_pred_lr)
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
matrix_ensemble = confusion_matrix(Y_test, Y_pred_ensemble)

with open("model.pkl", "wb") as file:
    pickle.dump(ensemble_model, file)
