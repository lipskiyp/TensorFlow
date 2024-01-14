"""
10-way classification of MNIST data set using a convolution network.
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy


# Collect data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()


# Preprocess data
train_data = train_data.reshape((60000, 28, 28, 1))  # (60000, 28, 28) -> (60000, 28, 28, 1)
train_data = train_data.astype('float32') / 255  # normalize

test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Model architecture
model = Sequential()
model.add(
    Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1))  # => (None, 26, 26, 32)
)
model.add(
    MaxPooling2D((2, 2))
)
model.add(
    Conv2D(64, (3, 3), activation="relu")
)
model.add(
    MaxPooling2D((2, 2))
)
model.add(
    Conv2D(64, (3, 3), activation="relu")
)
model.add(
    Flatten()  # classifier network accepts 1D vectors
)
model.add(
    Dense(64, activation="relu")
)
model.add(
    Dense(10, activation="softmax")  # 10-way classification
)


# Model compilation
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss=categorical_crossentropy,  # sparse_categorical_crossentropy if integer tensor is used for labels
    metrics=categorical_accuracy,  # i.e. hit ratio
)


# Train model
model.fit(
    train_data,
    train_labels,
    epochs=5,
    batch_size=64
)


test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_acc)