"""
Binary classification of imdb reviews (positive or negative).
"""

from keras.datasets import imdb
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

from utils import vectorize_sequences


num_words = 10000  # keep only top most frequently occurring words
num_validation_samples = 10000  # to separate validation data (to test model after every enum)

# Retrieve data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)  # data: lists of word indices (i.e. reviews), labels: list of 0s (-ve) and 1s (+ve) reviews

# Vectorize data
train_data, test_data = vectorize_sequences(train_data, dimensions=num_words), vectorize_sequences(test_data, dimensions=num_words)
train_labels, test_labels = np.asarray(train_labels).astype('float32'), np.asarray(test_labels).astype('float32')

# Separate validation set
validation_data, validation_labels = train_data[:num_validation_samples], train_labels[:num_validation_samples]
partial_train_data, partial_train_labels = train_data[num_validation_samples:], train_labels[num_validation_samples:]

# Model architecture
model = Sequential()
model.add(Dense(
    16, activation="relu", input_shape = (num_words,)
))
model.add(Dense(
    16, activation="relu"
))
model.add(Dense(
    1, activation="sigmoid",  # output as probability
))

# Model compilation
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss=binary_crossentropy,  # used to optimise model
    metrics=binary_accuracy,  # i.e. hit ratio
)

# Train model
history = model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=4,  # iterations over all samples in the x_train and y_train tensors
    batch_size=512,  # break data into small batches (iterate over all every epoch)
    validation_data=(  # tests model on validation data at the ned of every epoch
        validation_data,
        validation_labels
    )
)

results = model.evaluate(test_data, test_labels)  # [loss, binary_accuracy]


# Plot results
"""
history_dict = history.history
history_dict_keys = history_dict.keys() # => ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'] (one per metric)

loss_vals = history_dict["loss"]
accuracy_vals = history_dict["binary_accuracy"]
validation_loss_vals = history_dict["val_loss"]
validation_accuracy_vals = history_dict["val_binary_accuracy"]

epochs = range(1, len(loss_vals) + 1)

plt.plot(epochs, loss_vals, 'bo', label='Training loss')  # x-axis, y-axis, color, label
plt.plot(epochs, validation_loss_vals, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # clear figure

plt.plot(epochs, accuracy_vals, 'bo', label='Training acc')  # x-axis, y-axis, color, label
plt.plot(epochs, validation_accuracy_vals, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""