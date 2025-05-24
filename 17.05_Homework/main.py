import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Завантаження та підготовка даних Fashion MNIST
print("Завантаження Fashion MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Назви класів одягу
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Розмір тренувальної вибірки: {x_train.shape}")
print(f"Розмір тестової вибірки: {x_test.shape}")

# Нормалізація даних (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Розширення розмірності для CNN (додавання каналу)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Конвертація міток у категоріальний формат
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print("Дані підготовлено!")

# Створення згорткової нейронної мережі
def create_cnn_model():
    model = keras.Sequential([
        # Перший згортковий блок
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Другий згортковий блок
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Третій згортковий блок
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Повнозв'язні шари
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Створення моделі
model = create_cnn_model()

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Виведення архітектури моделі
print("\nАрхітектура CNN моделі:")
model.summary()

print(f"\nЗагальна кількість параметрів: {model.count_params():,}")

# Підрахунок параметрів по шарах
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

print(f"Тренувальні параметри: {trainable_params:,}")
print(f"Нетренувальні параметри: {non_trainable_params:,}")

# Callback для зменшення learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.0001
)

# Early stopping для запобігання перенавчанню
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\nПочаток тренування моделі...")

# Тренування моделі
history = model.fit(
    x_train, y_train_cat,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test_cat),
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

print("Тренування завершено!")

# Оцінка моделі
print("\nОцінка моделі на тестових даних:")
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Тестова точність: {test_accuracy:.4f}")
print(f"Тестова втрата: {test_loss:.4f}")

# Передбачення
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Детальний звіт класифікації
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Функція для візуалізації результатів тренування
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Графік точності
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Accuracy over Epochs', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Графік втрати
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Loss over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Функція для матриці плутанини
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Функція для візуалізації прикладів передбачень
def plot_predictions(x_test, y_test, y_pred_classes, class_names, num_examples=12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for i in range(num_examples):
        img = x_test[i].reshape(28, 28)
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred_classes[i]]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                         color='green' if y_test[i] == y_pred_classes[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Візуалізація результатів
print("\nПобудова графіків...")
plot_training_history(history)
plot_confusion_matrix(y_test, y_pred_classes, class_names)
plot_predictions(x_test, y_test, y_pred_classes, class_names)

# Детальна інформація про архітектуру
print("\nДетальна інформація про шари та фільтри:")

conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
for i, layer in enumerate(conv_layers):
    print(f"\nЗгортковий шар {i+1}:")
    print(f"  - Назва: {layer.name}")
    print(f"  - Кількість фільтрів: {layer.filters}")
    print(f"  - Розмір ядра: {layer.kernel_size}")
    print(f"  - Функція активації: {layer.activation.__name__}")
    print(f"  - Форма виходу: {layer.output.shape}")

# Збереження моделі
model.save('fashion_mnist_cnn_model.h5')
print("\nМодель збережена як 'fashion_mnist_cnn_model.h5'")

print("\n" + "="*60)
print("ПІДСУМОК РЕЗУЛЬТАТІВ:")
print("="*60)
print(f"Фінальна точність на тестових даних: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Фінальна втрата на тестовихданих: {test_loss:.4f}")
print(f"Кількість епох тренування: {len(history.history['accuracy'])}")
print(f"Загальна кількість параметрів моделі: {model.count_params():,}")
print("="*60)