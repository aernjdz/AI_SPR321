import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Завантаження даних
df = pd.read_csv("cars_realistic_with_brand.csv")
print(df.head())


X = df[['brand', 'model', 'year', 'engine_volume', 'mileage', 'horsepower']]  # features
y = df['price']  # target

print("Ознаки (features):", X.columns.tolist())
print("Ціль (target):", y.name)


categorical = ['brand', 'model']
numerical = ['year', 'engine_volume', 'mileage', 'horsepower']


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання моделі
model.fit(X_train, y_train)


your_car = pd.DataFrame([{
    'brand': 'Volkswagen',
    'model': 'Passat',
    'year': 2016,
    'engine_volume': 1.6,
    'mileage': 30,
    'horsepower': 210
}])

predicted_price = model.predict(your_car)
print(f"Прогнозована ціна автомобіля: {predicted_price[0]:,.2f} $")


y_pred = model.predict(X_test)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()
