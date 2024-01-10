"""
Single-label (i.e. mutually exclusive) multiclass classification of reuters newswires data.
"""

from keras.datasets import reuters
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

num_words = 10000  # top most frequently used words

(train_data, train_lables), (test_data, test_labels) = reuters.load_data()
word_index = reuters.get_word_index()  # can be used to decode sentences, word_inde[i-3] -> str


def vectorize_data(sequences, dimension=num_words):
    """
    Vectorize newswires' text data by converting integer list of words to binary vector of size dimension.
    """
    res = np.zeros((len(sequences), dimension))  # array of size: (number of sequences, number of words)
    for s, sequence in enumerate(sequences):  # for every newswire
        res[s, sequence] = 1  # 1 for every word that is found in the sequence
    return res

def to_one_hot(labels, dimesions=46):  # number of topics
    """
    Vectorize data to "one-hot" integer tensor (1 where index matches the topic) - common categorical encoding technique.

    from keras.utils.np_utils import to_categorical  -  same:
    train_label = to_categorical(train_label)
    """
    res = np.zeros((len(labels), dimesions))  # array of size: (number of sequences, number of topics)
    for l, label in enumerate(labels):
        res[l, label] = 1
    return res

train_data, test_data = vectorize_data(train_data), vectorize_data(test_data)  # converts raw data to a sequence of binary vectors
train_lables, test_labels = to_one_hot(train_lables), to_one_hot(test_labels)