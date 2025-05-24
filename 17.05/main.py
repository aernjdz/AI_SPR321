import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape input to 28x28x1 for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0


# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2),
    Dropout(0.25),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train_cnn, y_train_cat, epochs=20, validation_split=0.2, callbacks=[early_stop])

model.save_pretrained('num_cnn_model')

# test_loss, test_acc = model.evaluate(x_test, y_test_cat)
# print("Test accuracy:", test_acc)

import numpy as np

# Predict the first 5 test samples
predictions = model.predict(x_test_cnn[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])} - True: {y_test[i]}")
    plt.axis('off')
    plt.show()

# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np

# # --- Завантаження даних ---
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # --- Нормалізація та reshape ---
# x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
# x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

# # --- One-hot encoding ---
# y_train_cat = to_categorical(y_train, 10)
# y_test_cat = to_categorical(y_test, 10)

# # --- Розбиття на тренувальні і валідаційні дані ---
# x_train_part, x_val_part, y_train_part, y_val_part = train_test_split(
#     x_train_cnn, y_train_cat, test_size=0.2, random_state=42)

# # --- Аугментація даних ---
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )
# datagen.fit(x_train_part)

# # --- Побудова моделі ---
# model = Sequential([
#     Input(shape=(28,28,1)),
#     Conv2D(32, 3, activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(32, 3, activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2),
#     Dropout(0.25),

#     Conv2D(64, 3, activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(64, 3, activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2),
#     Dropout(0.25),

#     Flatten(),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # --- Callbacks ---
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# # --- Навчання моделі з аугментацією ---
# batch_size = 64
# history = model.fit(
#     datagen.flow(x_train_part, y_train_part, batch_size=batch_size),
#     epochs=50,
#     validation_data=(x_val_part, y_val_part),
#     callbacks=[early_stop, reduce_lr]
# )

# # --- Збереження моделі ---
# model.save_pretrained('saved_model.h5')


# # --- Оцінка на тесті ---
# test_loss, test_acc = model.evaluate(x_test_cnn, y_test_cat)
# print(f"Test accuracy: {test_acc:.4f}")

# # --- Візуалізація кількох прикладів з прогнозами ---
# predictions = model.predict(x_test_cnn[:10])
# for i in range(10):
#     plt.imshow(x_test[i], cmap='gray')
#     plt.title(f"Predicted: {np.argmax(predictions[i])} - True: {y_test[i]}")
#     plt.axis('off')
#     plt.show()
