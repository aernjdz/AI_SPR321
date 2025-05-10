import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    x = df['Area_m2'].values.reshape(-1, 1)
    y = df['Price_USD'].values.reshape(-1, 1)
    

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    
    return x_scaled, y_scaled, x_scaler, y_scaler


def build_model():
    model = keras.Sequential([
        layers.Dense(128, activation='tanh', input_shape=(1,)),
        layers.Dense(64, activation='tanh'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_model(model, x_train, y_train, epochs=500):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=30, 
        restore_best_weights=True
    )
    
    history = model.fit(
        x_train, 
        y_train, 
        validation_split=0.2, 
        epochs=epochs, 
        batch_size=16, 
        callbacks=[early_stopping], 
        verbose=1
    )
    return model, history


def visualize_results(x_train, y_train, x_test, nn_pred, lin_pred, x_scaler, y_scaler):

    x_train_orig = x_scaler.inverse_transform(x_train)
    y_train_orig = y_scaler.inverse_transform(y_train)
    x_test_orig = x_scaler.inverse_transform(x_test)
    nn_pred_orig = y_scaler.inverse_transform(nn_pred)
    lin_pred_orig = y_scaler.inverse_transform(lin_pred)
    
    plt.figure(figsize=(12, 7))
    plt.scatter(x_train_orig, y_train_orig, label="Training Data", alpha=0.6)
    plt.plot(x_test_orig, nn_pred_orig, label="Neural Network (tanh)", color="red", linewidth=2)
    plt.plot(x_test_orig, lin_pred_orig, label="Linear Regression", color="blue", linewidth=2)
    plt.title("House Price Prediction Comparison (with tanh)", fontsize=14)
    plt.xlabel("Area (m²)", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":

    x_train, y_train, x_scaler, y_scaler = load_and_prepare_data("house_prices_simple.csv")
    

    x_test = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
    

    model = build_model()
    model, history = train_model(model, x_train, y_train)
    nn_pred = model.predict(x_test)
    

    lin_model = LinearRegression()
    lin_model.fit(x_train, y_train)
    lin_pred = lin_model.predict(x_test)
    

    visualize_results(x_train, y_train, x_test, nn_pred, lin_pred, x_scaler, y_scaler)
    
 
    new_areas = np.random.uniform(min(x_train), max(x_train), 10).reshape(-1, 1)
    new_areas_orig = x_scaler.inverse_transform(new_areas)
    
    nn_prices = y_scaler.inverse_transform(model.predict(new_areas)).flatten()
    lin_prices = y_scaler.inverse_transform(lin_model.predict(new_areas)).flatten()
    
    print("\nPrediction Results:")
    print("Area (m²) | NN Price ($) | Linear Price ($)")
    print("-------------------------------------------")
    for area, nn, lin in zip(new_areas_orig.flatten(), nn_prices, lin_prices):
        print(f"{area:9.2f} | {nn:12.2f} | {lin:15.2f}")
    
    # Візуалізація історії навчання
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE (tanh)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss (tanh)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()