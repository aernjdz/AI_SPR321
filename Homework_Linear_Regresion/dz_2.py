import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("energy_usage_plus.csv")


X = df[['temperature', 'humidity', 'season', 'hour', 'district_type', 'is_weekend']]
y = df['consumption']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', ['temperature', 'humidity', 'hour', 'is_weekend']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['season', 'district_type'])
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


model.fit(X_train, y_train)

# Прогноз
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Справжнє споживання (кВт·год)")
plt.ylabel("Прогнозоване споживання (кВт·год)")
plt.title("Task 2: Справжнє vs Прогнозоване")
plt.grid(True)
plt.tight_layout()
plt.show()
