# Import Pandas and Numpy
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

df = pd.read_csv("data.csv") # Read dataset
pd.set_option("display.max_column", None)

Y = pd.DataFrame() # Initialize a new DataFrame
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

job_median = X['AGE'].median()
X['AGE'] = X['AGE'] / job_median

duration_median = X['DURATION'].median()
X['DURATION'] = X['DURATION'] / duration_median
X = X[X['DURATION'] < 2]
Y = Y.loc[X.index]  # Filter Y based on the indices of X after filtering
X["AMOUNT"] = normalize(X["AMOUNT"])

# Reset indices
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)

# Splitting traing and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train model
model_knn = KNeighborsClassifier(n_neighbors=6)
model_random_forest = RandomForestClassifier(max_depth=20, n_estimators=141)

model_knn.fit(X_train, Y_train)
model_random_forest.fit(X_train, Y_train)

Y_pred_knn = model_knn.predict(X_test)
Y_pred_random_forest = model_random_forest.predict(X_test)

matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
matrix_random_forest = confusion_matrix(Y_test, Y_pred_random_forest)

print(matrix_knn)
print(matrix_random_forest)
