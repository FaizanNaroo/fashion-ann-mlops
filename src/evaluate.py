import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = load_model("models/model.h5")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print("Evaluation completed successfully.")