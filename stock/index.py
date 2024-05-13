import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datetime import datetime

# Load the dataset
df = pd.read_csv("goldstock.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Extract date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Dayofweek'] = df['Date'].dt.dayofweek

# Features and target variable
X = df[['Year', 'Month', 'Day', 'Dayofweek', "Volume", "High", "Low", "Open"]]
Y = df['Close']  # Target variable: Close price

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, Y_train)
model_rf.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred_rf = model.predict(X_test)

# Calculate mean absolute error
mae = mean_squared_error(Y_test, Y_pred)
mae_rf = mean_squared_error(Y_test, Y_pred_rf)

print("Mean Absolute Error:", mae)
print("Mean Absolute Error RF:", mae_rf)

# Extract today's date
today = datetime.now().date()

# Extract date features for today's date
today_year = today.year
today_month = today.month
today_day = today.day
today_dayofweek = today.weekday()  # Monday is 0 and Sunday is 6

# Features for today's date
today_features = [[today_year, today_month, today_day, today_dayofweek, 177899, 2370.80, 2337.60, 2369.10]]
today_gold_price = model.predict(today_features)
today_gold_price_rf = model_rf.predict(today_features)
print("Predicted gold price for today:", today_gold_price[0])
print("Predicted gold price for today:", today_gold_price_rf[0])
