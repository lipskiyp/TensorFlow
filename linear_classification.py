"""
(Binary) Linear Classification Algorithm. (UNFINISHED)
"""

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. Load Data
data = load_breast_cancer() # type(data) = sklearn.utils.Bunch

# 2. Split Data: train and test (validation)
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.33
)

n, d = x_train.shape

scaler = StandardScaler() # standardises the data (i.e. mean = 0 and sigma = 1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3. Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(d,)), # Determines the size of the input vector x. NB model can take any number of samples hence, n is not required
    tf.keras.layers.Dense(1, activation="sigmoid") # 1 determines the output size. logistic regression => sigmoid activation function
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy", # optimisation metric (distance from regression line)
    metrics=["accuracy"] # correct / total
)

# 4. Fit Model
r = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100 # number of MC loops
)

plt.plot(r.history["loss"], label="Loss")
plt.plot(r.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()
