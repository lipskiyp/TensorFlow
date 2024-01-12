"""
Regression of Boston housing price data.
"""

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mse
from keras.metrics import mae
from keras.optimizers import RMSprop
import numpy as np


# Collect data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()  # (404, 13), (102, 13)


# Feature-wise data normalization (to optimize learning difficulty)
train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)
test_data = (test_data - test_data.mean(axis=0)) / test_data.std(axis=0)


# Build model
def build_model():  # to call same model multiple times
    # Model architecture
    model = Sequential()
    model.add(
        Dense(64, activation="relu", input_shape=(train_data.shape[1],))
    )
    model.add(
        Dense(64, activation="relu")
    )
    model.add(
        Dense(1)  # linear layer (network can predict values in any range)
    )

    # Model compilation
    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss=mse,  # mean squared error E[(Prediction - Target)^2]
        metrics=mae,  # mean absolute error (Prediction - Target)
    )

    return model


# K-Fold Cross Validation
num_partitions = 4  # K
num_validation_samples = len(train_data) // 4  # split data into K-partitions
num_epochs=80
all_history = []

for k in range(num_partitions):  # for every partitions (fold)
    validation_data = train_data[k * num_validation_samples: (k+1) * num_validation_samples]
    validation_targets = train_targets[k * num_validation_samples: (k+1) * num_validation_samples]

    partial_train_data = np.concatenate([
        train_data[:k * num_validation_samples],
        train_data[(k+1) * num_validation_samples:],
    ], axis=0)
    partial_train_targets = np.concatenate([
        train_targets[:k * num_validation_samples],
        train_targets[(k+1) * num_validation_samples:],
    ], axis=0)

    model = build_model()  # instantiate K identical models

    history = model.fit(  # train each model on K-1 partitions
        partial_train_data,
        partial_train_targets,
        epochs=num_epochs,
        batch_size=1,
        verbose=0,  #  silent mode
        validation_data=(
            validation_data, validation_targets  # validate on the remaining partition
        )
    )

    mae_history = history.history["val_mean_absolute_error"]
    all_history.append(mae_history)


average_mae_history = [np.mean([x[n] for x in all_history]) for n in range(num_epochs)]


# Once the model is tuned (e.g. epochs set to 80) train model on all of the training model.
model = build_model()
history = model.fit(  # train each model on K-1 partitions
    train_data,
    train_targets,
    epochs=num_epochs,
    batch_size=16,
    verbose=0,  #  silent mode
)

print("hey")
