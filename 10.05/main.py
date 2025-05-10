import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression

def load_data_from_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    x = df['Area_m2'].values.reshape(-1, 1)
    y = df['Price_USD'].values.reshape(-1, 1)
    return x, y

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, x_train, y_train, epochs=500):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
              batch_size=8, callbacks=[early_stopping], verbose=0)
    return model

def visualize_results(x_train, y_train, x_test, nn_pred, lin_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.4)
    plt.plot(x_test, nn_pred, color="red")
    plt.plot(x_test, lin_pred, color="blue")
    plt.xlabel("Area (m²)")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    x_train, y_train = load_data_from_csv("house_prices_simple.csv")
    x_test = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)

    model = build_model()
    train_model(model, x_train, y_train)
    nn_pred = model.predict(x_test)

    lin_model = LinearRegression()
    lin_model.fit(x_train, y_train)
    lin_pred = lin_model.predict(x_test)

    visualize_results(x_train, y_train, x_test, nn_pred, lin_pred)

    new_areas = np.random.uniform(min(x_train), max(x_train), 10).reshape(-1, 1)
    nn_prices = model.predict(new_areas).flatten()
    lin_prices = lin_model.predict(new_areas).flatten()

    print("Area (m²) | NN Price ($) | Linear Price ($)")
    print("-------------------------------------------")
    for area, nn, lin in zip(new_areas.flatten(), nn_prices, lin_prices):
        print(f"{area:9.2f} | {nn:12.2f} | {lin:15.2f}")
