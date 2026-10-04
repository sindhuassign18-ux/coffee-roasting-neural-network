# ============================================================
# Coffee Roasting Classification using a Neural Network
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------

X = np.array([
    [180, 14.5], [190, 14.0], [200, 13.5], [210, 13.0],
    [220, 12.5], [230, 12.0], [240, 11.8], [250, 11.5],
    [180, 11.5], [190, 11.8], [200, 15.0], [210, 15.5],
    [220, 16.0], [230, 16.0], [240, 15.5], [250, 15.0]
])

# 1 = Good Roast, 0 = Bad Roast
Y = np.array([
    [1], [1], [1], [1],
    [1], [1], [1], [0],
    [0], [0], [0], [0],
    [0], [0], [0], [0]
])

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ------------------------------------------------------------
# 2. Visualize the Dataset
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X[Y[:, 0] == 1, 0],
    X[Y[:, 0] == 1, 1],
    marker='o',
    label='Good Roast'
)

plt.scatter(
    X[Y[:, 0] == 0, 0],
    X[Y[:, 0] == 0, 1],
    marker='x',
    label='Bad Roast'
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Duration (minutes)")
plt.title("Coffee Roasting Dataset")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# 3. Normalize the Input Data
# ------------------------------------------------------------

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

Xn = (X - mean) / std


# ------------------------------------------------------------
# 4. Create the Neural Network
# ------------------------------------------------------------

tf.random.set_seed(1234)

model = Sequential([
    Dense(3, activation='sigmoid', input_shape=(2,), name='layer1'),
    Dense(1, activation='sigmoid', name='layer2')
])

model.summary()


# ------------------------------------------------------------
# 5. Compile the Model
# ------------------------------------------------------------

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)


# ------------------------------------------------------------
# 6. Train the Model
# ------------------------------------------------------------

history = model.fit(
    Xn,
    Y,
    epochs=500,
    verbose=1
)


# ------------------------------------------------------------
# 7. Display Learned Weights and Biases
# ------------------------------------------------------------

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()

print("\nLayer 1 Weights:\n", W1)
print("\nLayer 1 Bias:\n", b1)
print("\nLayer 2 Weights:\n", W2)
print("\nLayer 2 Bias:\n", b2)


# ------------------------------------------------------------
# 8. Plot Training Loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()


# ------------------------------------------------------------
# 9. Test the Model with New Data
# ------------------------------------------------------------

X_test = np.array([
    [200, 13.5],
    [200, 16.0],
    [240, 12.0],
    [180, 11.0]
])

# Use the same mean and standard deviation from training data
X_test_normalized = (X_test - mean) / std

# Make predictions
predictions = model(X_test_normalized, training=False).numpy()


# ------------------------------------------------------------
# 10. Convert Probabilities into Class Predictions
# ------------------------------------------------------------

decisions = (predictions >= 0.5).astype(int)

print("\nPrediction Results:")

for i in range(len(X_test)):

    temperature = X_test[i, 0]
    duration = X_test[i, 1]
    probability = predictions[i, 0]

    result = "GOOD ROAST" if decisions[i, 0] == 1 else "BAD ROAST"

    print(
        f"Temperature: {temperature}°C | "
        f"Duration: {duration} min | "
        f"Probability: {probability:.4f} | "
        f"Result: {result}"
    )


# ------------------------------------------------------------
# 11. Visualize Predictions
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Training data
plt.scatter(
    X[Y[:, 0] == 1, 0],
    X[Y[:, 0] == 1, 1],
    marker='o',
    label='Good Training Data'
)

plt.scatter(
    X[Y[:, 0] == 0, 0],
    X[Y[:, 0] == 0, 1],
    marker='x',
    label='Bad Training Data'
)

# Test predictions
for i in range(len(X_test)):

    marker = '*' if decisions[i, 0] == 1 else 's'

    plt.scatter(
        X_test[i, 0],
        X_test[i, 1],
        marker=marker,
        s=200,
        label='Test Prediction' if i == 0 else ""
    )

plt.xlabel("Temperature (°C)")
plt.ylabel("Duration (minutes)")
plt.title("Coffee Roasting Neural Network Predictions")
plt.legend()
plt.grid()
plt.show()

