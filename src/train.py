import csv
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

dense_units = params["train"]["dense_units"]
dropout_rate = params["train"]["dropout_rate"]
learning_rate = params["train"]["learning_rate"]
epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(dense_units, activation="relu"),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size
)

Path("models").mkdir(parents=True, exist_ok=True)

model.save("models/model.h5")

with open("models/history.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(history.history.keys())
    writer.writerows(zip(*history.history.values()))

print("Model training completed successfully.")