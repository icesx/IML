# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

from tensorflow import keras
import tensorflow as tf
from datasets.load_data import keras_load_data


def load_mnist():
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.mnist)
    print(train_images.shape)
    return train_images, test_images, train_labels, test_labels


def run_module(train_images, train_lables):
    module = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(120, activation="relu"),
        keras.layers.Dense(10)
    ])
    module.compile(optimizer="adam",
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    module.summary()
    module.fit(train_images, train_lables, epochs=5)
    return module


def predict(module, test_images, test_lables):
    predic_module = keras.Sequential([module, keras.layers.Softmax()])
    prediced = predic_module.predict(test_images)
    print(test_lables[0], tf.argmax(prediced[0]))


if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels = load_mnist()
    module = run_module(train_images, train_labels)
    predict(module, test_images, test_labels)
