
from pathlib import Path
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/skin_non_skin_model.keras")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Final layer:", model.layers[-1].name)
print("Final activation:",
      getattr(model.layers[-1], "activation", None))

print("\nModel summary:")
model.summary()

print("\nTesting zero and random inputs:")
height = model.input_shape[1]
width = model.input_shape[2]

for name, image in [
    ("zeros_0_255", np.zeros((1, height, width, 3), dtype=np.float32)),
    ("random_0_255", np.random.uniform(
        0, 255, (1, height, width, 3)
    ).astype(np.float32)),
    ("random_0_1", np.random.uniform(
        0, 1, (1, height, width, 3)
    ).astype(np.float32)),
]:
    result = model.predict(image, verbose=0)
    print(name, "=>", result)
