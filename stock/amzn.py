import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime

# Load the data
df = pd.read_csv("AMZN.csv")  # Replace "AMZN.csv" with the path to your data file

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Dayofweek'] = df['date'].dt.dayofweek

# Define features and target
X = df[['Year', 'Month', 'Day', 'Dayofweek', 'open']]
y_open = df['open']
y_high = df['high']
y_low = df['low']
y_close = df['close']

# Split data into train and test sets
X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.2, random_state=42)
X_train, X_test, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.2, random_state=42)
X_train, X_test, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.2, random_state=42)


# Initialize and fit the model for high prices
model_high = LinearRegression()
model_high.fit(X_train, y_high_train)

# Initialize and fit the model for low prices
model_low = LinearRegression()
model_low.fit(X_train, y_low_train)

# Initialize and fit the model for close prices
model_close = LinearRegression()
model_close.fit(X_train, y_close_train)

# Predictions
# Extract today's date
today = datetime(2024, 5, 10)

# Extract date features for today's date
today_year = today.year
today_month = today.month
today_day = today.day
today_dayofweek = today.weekday()  # Monday is 0 and Sunday is 6

predicted_high = model_high.predict([[today_year, today_month, today_day, today_dayofweek, 189.16]])[0]
predicted_low = model_low.predict([[today_year, today_month, today_day, today_dayofweek, 189.16]])[0]
predicted_close = model_close.predict([[today_year, today_month, today_day, today_dayofweek, 189.16]])[0]

print("Predicted High Price:", predicted_high)
print("Predicted Low Price:", predicted_low)
print("Predicted Close Price:", predicted_close)

# Evaluate the models
y_high_pred_test = model_high.predict(X_test)
y_low_pred_test = model_low.predict(X_test)
y_close_pred_test = model_close.predict(X_test)

mse_high = mean_squared_error(y_high_test, y_high_pred_test)
mse_low = mean_squared_error(y_low_test, y_low_pred_test)
mse_close = mean_squared_error(y_close_test, y_close_pred_test)

print("\nModel Evaluation:")
print("High Price Mean Squared Error:", mse_high)
print("Low Price Mean Squared Error:", mse_low)
print("Close Price Mean Squared Error:", mse_close)
