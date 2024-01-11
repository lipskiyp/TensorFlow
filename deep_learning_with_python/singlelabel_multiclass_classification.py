"""
Single-label (i.e. mutually exclusive) multiclass classification of reuters newswires data.
"""

from keras.datasets import reuters
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

from utils import vectorize_sequences, to_one_hot


num_words = 10000  # top most frequently used words
num_validation_samples = 1000  # to separate validation data (to test model after every enum)


# Collect data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)
word_index = reuters.get_word_index()  # can be used to decode sentences, word_inde[i-3] -> str


# Vectorize data
train_data, test_data = vectorize_sequences(train_data, dimensions=num_words), vectorize_sequences(test_data, dimensions=num_words)  # converts raw data to a sequence of binary vectors
train_labels, test_labels = to_one_hot(train_labels, dimensions=46), to_one_hot(test_labels, dimensions=46)  # 46 = number of topics
# Alternatively use integer tensor: train_data, test_labels = np.array(train_labels), np.array(test_labels)
# with "sparse_categorical_crossentropy" loss function (instead of "categorical_crossentropy").

# Separate validation data
partial_train_data, validation_data = train_data[num_validation_samples:], train_data[:num_validation_samples]
partial_train_labels, validation_labels = train_labels[num_validation_samples:], train_labels[:num_validation_samples]


# Model architecture
# NB No layer should have a hidden unit that is less than the dimension of the classifier to avoid loosing information
# i.e. output of each layer must have at least 46 hidden units (1 for every topic)
model = Sequential()
model.add(
    Dense(64, activation="relu", input_shape=(num_words,)),  # NB excludes samples axis
)
model.add(
    Dense(64, activation="relu"),
)
model.add(
    Dense(46, activation="softmax"),  # probability distribution vector across all topics (sum to 1)
)


# Model compilation
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss=categorical_crossentropy,  # sparse_categorical_crossentropy if integer tensor is used for labels
    metrics=categorical_accuracy,  # i.e. hit ratio
)


# Train model
history = model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=9,  # iterations over all samples in the x_train and y_train tensors
    batch_size=512,  # break data into small batches (iterate over all every epoch)
    validation_data=(  # tests model on validation data at the end of every epoch
        validation_data,
        validation_labels
    )
)

results = model.evaluate(test_data, test_labels)  # [loss, categorical_accuracy]


# Plot results
history_dict = history.history
history_dict_keys = history_dict.keys() # => ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'] (one per metric)

loss_vals = history_dict["loss"]
accuracy_vals = history_dict["categorical_accuracy"]
validation_loss_vals = history_dict["val_loss"]
validation_accuracy_vals = history_dict["val_categorical_accuracy"]

epochs = range(1, len(loss_vals) + 1)

plt.plot(epochs, loss_vals, 'bo', label='Training loss')  # x-axis, y-axis, color, label
plt.plot(epochs, validation_loss_vals, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
