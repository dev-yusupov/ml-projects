from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.utils import resample
import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv("data.csv")
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

# Normalize 'AMOUNT'
X["AMOUNT"] = normalize(X[["AMOUNT"]])

# Scale 'AMOUNT' using Min-Max scaling
scaler = MinMaxScaler()
X["AMOUNT"] = scaler.fit_transform(X[["AMOUNT"]])

# Calculate median for 'AGE' and 'DURATION'
job_median = X['AGE'].median()
X['AGE'] = X['AGE'] / job_median

duration_median = X['DURATION'].median()
X['DURATION'] = X['DURATION'] / duration_median
X = X[X['DURATION'] < 2]
Y = Y.loc[X.index]  # Filter Y based on the indices of X after filtering

# Reset indices
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)

# Splitting training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Get the number of samples in the majority class
num_majority_samples = sum(Y_train['Returned'] == 0)

# If the number of samples in the majority class is greater than 300, downsample to 300
if num_majority_samples > 300:
    # Downsample the majority class using RandomUnderSampler
    X_train_resampled, Y_train_resampled = resample(X_train[Y_train['Returned'] == 0],
                                                    Y_train[Y_train['Returned'] == 0],
                                                    replace=False,
                                                    n_samples=300,
                                                    random_state=42)
else:
    # If the number of samples is less than 300, use all available samples
    X_train_resampled = X_train[Y_train['Returned'] == 0]
    Y_train_resampled = Y_train[Y_train['Returned'] == 0]

# Combine the downsampled majority class with the minority class
X_train_balanced = pd.concat([X_train_resampled, X_train[Y_train['Returned'] == 1]])
Y_train_balanced = pd.concat([Y_train_resampled, Y_train[Y_train['Returned'] == 1]])

# Train models
model_knn = KNeighborsClassifier(n_neighbors=6)
model_random_forest = RandomForestClassifier(max_depth=50, n_estimators=191)
model_tree = DecisionTreeClassifier(max_depth=30)
model_gradient_boosting = GradientBoostingClassifier()
ensemble_clf = VotingClassifier(estimators=[
    ('knn', model_knn),
    ('random_forest', model_random_forest),
    ('decision_tree', model_tree),
    ('gradient_boosting', model_gradient_boosting)
], voting='hard')

model_knn.fit(X_train_balanced, Y_train_balanced)
model_random_forest.fit(X_train_balanced, Y_train_balanced)
model_tree.fit(X_train_balanced, Y_train_balanced)
model_gradient_boosting.fit(X_train_balanced, Y_train_balanced)
ensemble_clf.fit(X_train_balanced, Y_train_balanced)

# Make predictions
Y_pred_knn = model_knn.predict(X_test)
Y_pred_random_forest = model_random_forest.predict(X_test)
Y_pred_tree = model_tree.predict(X_test)
Y_pred_gradient_boosting = model_gradient_boosting.predict(X_test)
Y_pred_ensemble = ensemble_clf.predict(X_test)

# Evaluate models
matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
matrix_random_forest = confusion_matrix(Y_test, Y_pred_random_forest)
matrix_tree = confusion_matrix(Y_test, Y_pred_tree)
matrix_gradient_boosting = confusion_matrix(Y_test, Y_pred_gradient_boosting)
matrix_ensemble = confusion_matrix(Y_test, Y_pred_ensemble)

print("Confusion matrix for KNN:\n", matrix_knn)
print("Confusion matrix for Random Forest:\n", matrix_random_forest)
print("Confusion matrix for Decision Tree:\n", matrix_tree)
print("Confusion matrix for Gradient Boosting:\n", matrix_gradient_boosting)
print("Confusion matrix for Ensemble:\n", matrix_ensemble)
