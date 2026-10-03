import csv
import numpy as np
import tensorflow as tf

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

model.save("models/model.h5")

with open("models/history.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(history.history.keys())
    writer.writerows(zip(*history.history.values()))

print("Model training completed successfully.")