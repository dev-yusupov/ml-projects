import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize 
df = pd.read_csv("data.csv")
cols = ['CHK_ACCT', 'DURATION', 'HISTORY', 'AMOUNT', 'SAV_ACCT', 'EMPLOYMENT', 'INSTALL_RATE', 'PRESENT_RESIDENT', 'REAL_ESTATE', 'AGE', 'NUM_CREDITS', 'JOB']

cols_to_drop = ['OBS#', 'NEW_CAR', 'USED_CAR', 'FURNITURE', 'RADIO/TV', 'EDUCATION', 'RETRAINING', 'MALE_DIV', 'MALE_SINGLE', 'MALE_MAR_or_WID', 'CO-APPLICANT', 
                'GUARANTOR', 'PROP_UNKN_NONE', 'OTHER_INSTALL', 'RENT', 'OWN_RES', 'NUM_DEPENDENTS', 'TELEPHONE', 'FOREIGN']
df = df.drop(cols_to_drop, axis=1)
df = df.dropna(axis = 0)

Y = pd.DataFrame()
Y["answer"] = df["RESPONSE"]
df = df.drop(["RESPONSE"], axis = 1)

df_norm = normalize(df)
df = pd.DataFrame(df_norm, columns = cols)
for col in df.columns:
    Y = Y[df[col] < 1.5]
    df = df[df[col] < 1.5]

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.14)
print("LR:")
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
Y_pred_lr = model_lr.predict(X_test)
matrix_lr = confusion_matrix(Y_test, Y_pred_lr)
print(matrix_lr)

print("Rf")
model_rf = RandomForestClassifier(n_estimators = 1001, max_depth = 3)
model_rf.fit(X_train, Y_train)
Y_pred_rf = model_rf.predict(X_test)
matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
print(matrix_rf)

print("DT:")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, Y_train)
Y_pred_dt = model_dt.predict(X_test)
matrix_dt = confusion_matrix(Y_test, Y_pred_dt)
print(matrix_dt)

model_knn = KNeighborsClassifier(n_neighbors = 2)
model_knn.fit(X_train, Y_train)
Y_pred_knn = model_knn.predict(X_test)
print("KNN:")
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
print(matrix_knn)
