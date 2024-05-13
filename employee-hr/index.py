import pandas as pd
import pickle as pkl

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the synthetic dataset for training the model
df_train = pd.read_csv("validated_data.csv").dropna().reset_index()

# Columns to use in training
columns = ['last_evaluation', 'average_montly_hours', 'time_spend_company',
           'promotion_last_5years', 'dept', 'salary']

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode categorical variables
df_train["dept"] = label_encoder.fit_transform(df_train["dept"])
df_train["salary"] = label_encoder.fit_transform(df_train["salary"])

# Prepare features (X) and target variable (Y) for training
X_train = df_train[columns]
Y_train = df_train["satisfaction_level"]

# Split the data into training and testing sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()
model_tree = GradientBoostingRegressor()

# Fit the model to the training data
model.fit(X_train, Y_train)
model_tree.fit(X_train, Y_train)

# Predict on the validation data
Y_pred_val = model.predict(X_val)
# Predict on the validation data for the Gradient Boosting model
Y_pred_tree_val = model_tree.predict(X_val)

# Calculate evaluation metrics on the validation data for the Gradient Boosting model
mae_tree_val = mean_absolute_error(Y_val, Y_pred_tree_val)
mse_tree_val = mean_squared_error(Y_val, Y_pred_tree_val)

print("MAE on GB validation data:", mae_tree_val)
print("MSE on GB validation data:", mse_tree_val)


# Load the testing dataset
df_test = pd.read_csv("data.csv").dropna().reset_index()

# Apply the same preprocessing steps as the training dataset
df_test["dept"] = label_encoder.fit_transform(df_test["dept"])
df_test["salary"] = label_encoder.fit_transform(df_test["salary"])

# Prepare features (X) and target variable (Y) for testing
X_test = df_test[columns]
Y_test = df_test["satisfaction_level"]

# Predict on the testing data for the Gradient Boosting model
Y_pred_tree_test = model_tree.predict(X_test)

# Calculate evaluation metrics on the testing data for the Gradient Boosting model
mae_tree_test = mean_absolute_error(Y_test, Y_pred_tree_test)
mse_tree_test = mean_squared_error(Y_test, Y_pred_tree_test)

print("MAE on testing data for GB model:", mae_tree_test)
print("MSE on testing data for GB model:", mse_tree_test)


with open("employee.pkl", 'wb') as file:
    pkl.dump(model_tree, file=file)