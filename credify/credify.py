from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import normalize, MinMaxScaler, PolynomialFeatures
import pickle
import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv("synthetic_data.csv")
pd.set_option("display.max_column", None)

# Initialize Y
Y = pd.DataFrame()
Y["Returned"] = df['RESPONSE']

# Columns of dataset
columns = ['OBS#', 'CHK_ACCT', 'DURATION', 'HISTORY', 'NEW_CAR', 'USED_CAR',
       'FURNITURE', 'RADIO/TV', 'EDUCATION', 'RETRAINING', 'AMOUNT',
       'SAV_ACCT', 'EMPLOYMENT', 'INSTALL_RATE', 'MALE_DIV', 'MALE_SINGLE',
       'MALE_MAR_or_WID', 'CO-APPLICANT', 'GUARANTOR', 'PRESENT_RESIDENT',
       'REAL_ESTATE', 'PROP_UNKN_NONE', 'AGE', 'OTHER_INSTALL', 'RENT',
       'OWN_RES', 'NUM_CREDITS', 'JOB', 'NUM_DEPENDENTS', 'TELEPHONE', 'FOREIGN', 'RESPONSE']

# Initialize X
X = pd.DataFrame()
X["ACCOUNT_STATUS"] = df['CHK_ACCT']
X["DURATION"] = df["DURATION"]
X["HISTORY"] = df["HISTORY"]
X["EDUCATION"] = df["EDUCATION"]
X["AMOUNT"] = df["AMOUNT"]
X["EMPLOYMENT"] = df["EMPLOYMENT"]
X["GUARANTOR"] = df["GUARANTOR"]
X["AGE"] = df["AGE"]
X["JOB"] = df["JOB"]
X['NUM_CREDITS'] = df['NUM_CREDITS']
X['REAL_ESTATE'] = df['REAL_ESTATE']

# Normalize 'AMOUNT'
X["AMOUNT"] = normalize(X[["AMOUNT"]])

# Scale 'AMOUNT' using Min-Max scaling
scaler = MinMaxScaler()
X["AMOUNT"] = scaler.fit_transform(X[["AMOUNT"]])
X["AGE"] = scaler.fit_transform(X[["AGE"]])
X["DURATION"] = scaler.fit_transform(X[["DURATION"]])

# Reset indices
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)

# Splitting training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train models
model_knn = KNeighborsClassifier(n_neighbors=6)
model_random_forest = RandomForestClassifier(max_depth=50, n_estimators=191)
model_tree = DecisionTreeClassifier(max_depth=30)
model_logistic = LogisticRegression()
model_gradient_boosting = GradientBoostingClassifier()
model_svc = SVC()
model_ada = AdaBoostClassifier()
ensemble_clf = VotingClassifier(estimators=[
    ('knn', model_knn),
    ('random_forest', model_random_forest),
    ('decision_tree', model_tree),
    ('gradient_boosting', model_gradient_boosting),
    ('logistic', model_logistic),
    ("svc", model_svc),
    ("ada", model_ada)
], voting='hard')

model_knn.fit(X_train, Y_train)
model_random_forest.fit(X_train, Y_train)
model_tree.fit(X_train, Y_train)
model_gradient_boosting.fit(X_train, Y_train)
ensemble_clf.fit(X_train, Y_train)
model_logistic.fit(X_train, Y_train)
model_svc.fit(X_train, Y_train)
model_ada.fit(X_train, Y_train)

# Make predictions
Y_pred_knn = model_knn.predict(X_test)
Y_pred_random_forest = model_random_forest.predict(X_test)
Y_pred_tree = model_tree.predict(X_test)
Y_pred_gradient_boosting = model_gradient_boosting.predict(X_test)
Y_pred_ensemble = ensemble_clf.predict(X_test)
Y_pred_logistic = model_logistic.predict(X_test)
Y_pred_svc = model_svc.predict(X_test)
Y_pred_ada = model_ada.predict(X_test)

# Evaluate models
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
matrix_random_forest = confusion_matrix(Y_test, Y_pred_random_forest)
matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
matrix_gradient_boosting = confusion_matrix(Y_test, Y_pred_gradient_boosting)
matrix_ensemble = confusion_matrix(Y_test, Y_pred_ensemble)
matrix_logistic = confusion_matrix(Y_test, Y_pred_logistic)
matrix_svc = confusion_matrix(Y_test, Y_pred_svc)
matrix_ada = confusion_matrix(Y_test, Y_pred_ada)

print("Confusion matrix for KNN:\n", matrix_knn)
print("Confusion matrix for Random Forest:\n", matrix_random_forest)
print("Confusion matrix for Decision Tree:\n", matrix_tree)
print("Confusion matrix for Gradient Boosting:\n", matrix_gradient_boosting)
print("Confusion matrix for Ensemble:\n", matrix_ensemble)
print("Confusion matrix for Logistic:\n", matrix_logistic)
print("Confusion matrix for SVC:\n", matrix_svc)
print("Confusion matrix for Ada:\n", matrix_ada)


with open("model.pkl", 'wb') as model:
    pickle.dump(model_gradient_boosting, model)