import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('fuel_consumption_vs_speed.csv')  

X = df['speed_kmh'].values.reshape(-1, 1)
y = df['fuel_consumption_l_per_100km'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 4, 5]
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append((degree, mse, mae, model, poly))

best_model_info = min(results, key=lambda x: x[1])  # мінімальний MSE
best_degree, best_mse, best_mae, best_model, best_poly = best_model_info

print(f"Найкращий ступінь полінома: {best_degree}")
print(f"MSE: {best_mse:.4f}, MAE: {best_mae:.4f}")

# 3. Прогнози для 35, 95, 140 км/год
speeds = np.array([35, 95, 140]).reshape(-1, 1)
speeds_poly = best_poly.transform(speeds)
predicted_consumption = best_model.predict(speeds_poly)

for speed, consumption in zip(speeds.flatten(), predicted_consumption):
    print(f"Прогноз витрат пального на швидкості {speed} км/год: {consumption:.2f} л/100км")

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = best_poly.transform(X_plot)
y_plot = best_model.predict(X_plot_poly)

plt.scatter(X, y, color='red', label='Дані')
plt.plot(X_plot, y_plot, label=f'Поліноміальна регресія (ст. {best_degree})')
plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрати пального (л/100км)')
plt.legend()
plt.show()
