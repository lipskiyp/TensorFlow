"""
Utility functions.
"""

import numpy as np


def vectorize_sequences(sequences, dimensions):
    res = np.zeros((len(sequences), dimensions))
    for s, sequence in enumerate(sequences):
        res[s, sequence] = 1  # convert each review to vector of size num_words (1 if word with corresponding index is present in a review)
    return res


def to_one_hot(labels, dimensions):
    """
    Vectorize data to "one-hot" integer tensor (1 where index matches the topic) - common categorical encoding technique.

    from keras.utils.np_utils import to_categorical  -  same:
    train_label = to_categorical(train_label)
    """
    res = np.zeros((len(labels), dimensions))  # array of size: (number of sequences, number of topics)
    for l, label in enumerate(labels):
        res[l, label] = 1
    return res
