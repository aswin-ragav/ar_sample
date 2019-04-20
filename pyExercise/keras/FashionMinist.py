'''
    Basic classification problem
'''

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# URL:- https://www.tensorflow.org/tutorials/keras/basic_classification
class FashionMinist:

    # Description:- Import the Fashion MNIST dataset
    def download_Fashion_MINIST_dataset(self):
        return keras.datasets.fashion_mnist

    # manual entered class names
    def get_class_names(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag',  'Ankle boot']

    # Build the Sequential model
    def build_Sequential_model(self):
        return keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    # Compile the model
    def compile_model(self):
        return model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    
    # Train the model
    def train_model(self, train_images, train_labels):
        return model.fit(train_images, train_labels, epochs=5)


    # Evaluate accuracy
    def evaluate_accuracy(self, test_images, test_labels):
        return model.evaluate(test_images, test_labels)


    # Make predictions
    def start(self, test_images):
        return model.predict(test_images)


fm = FashionMinist() 
fashion_mnist = fm.download_Fashion_MINIST_dataset()

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

#print("train image :-  ", train_images[0])
#print("train label :-  ", train_labels[0])
#print("test image :-  ", test_images[0])
#print("test label :-  ", test_labels[0])

train_images = train_images / 255.0

test_images = test_images / 255.

model = fm.build_Sequential_model()

fm.compile_model()
fm.train_model(train_images, train_labels)

test_loss, test_acc = fm.evaluate_accuracy(test_images, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

predictions = fm.start(test_images)

print(predictions[0])


