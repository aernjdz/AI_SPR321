import tensorflow as tf
import tf2onnx

# Завантаження моделі
seq_model = tf.keras.models.load_model('num_cnn_model_improved.h5')

# Обгортання у функціональну модель (Model), щоб уникнути помилки з .output_names
model = tf.keras.Model(inputs=seq_model.input, outputs=seq_model.output)

# Вказуємо форму входу — наприклад, для MNIST:
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

# Конвертація
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Збереження
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
