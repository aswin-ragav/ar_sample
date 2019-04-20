'''
    Text classification

    This is an example of binary—or two-class—classification, an important and widely applicable kind of machine learning problem.
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np


class Imdb:

    def download_Imdb_dataset(self):
        return keras.datasets.imdb

    def decode_review(self, text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])



im = Imdb()
imdb = im.download_Imdb_dataset()

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
'''
    Convert the integers back to words
'''
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# show review by index
#print(im.decode_review(train_data[0]))

# Prepare the data
'''
 we can pad the arrays so they all have the same length, 
 then create an integer tensor of shape max_length * num_reviews. 
 We can use an embedding layer capable of handling this shape as 
 the first layer in our network.
'''
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(train_data[0])

# Build the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# Loss function and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)

print(results)

# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()







