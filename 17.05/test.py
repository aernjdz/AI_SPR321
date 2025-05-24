from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Load model ===
model = load_model("saved_model.h5")

# === Step 2: Load custom images ===
img_dir = "numbers"
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for image_file in sorted(image_files):  # Sorted for consistent order
    img_path = os.path.join(img_dir, image_file)

    # Load, convert to grayscale, resize to 28x28
    img = Image.open(img_path).convert("L").resize((28, 28))

    # Invert if background is white
    if np.mean(img) > 127:
        img = np.invert(img)

    # Convert to array and normalize
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # === Step 3: Predict ===
    prediction = model.predict(img_array)[0]
    predicted_label = np.argmax(prediction)

    # === Step 4: Show result and prediction chart ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Show input image
    ax1.imshow(img_array[0].reshape(28, 28), cmap="gray")
    ax1.set_title(f"{image_file} → Predicted: {predicted_label}")
    ax1.axis("off")

    # Show probability bar chart
    ax2.bar(range(10), prediction, color='gray')
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Confidence")
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()
