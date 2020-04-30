# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    # load data
    mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # check shape
    X_train.shape, y_train.shape

    # definig class name
    class_name = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                  'shirt', 'sneaker', 'bag', 'ankle boot']
    # changing scale 0-255 to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # building model
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten, Dense

    model = Sequential()
    # we can pass an array in sequential model or just add another layer
    model.add(Flatten(input_shape=(28, 28)))
    # flatten is used to change input data in 1d
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    # compilation of our model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=2)

    # chechking test_loss, test_acc
    test_loss, test_acc = model.evaluate(X_test, y_test)
    test_loss, test_acc

    # accuracy
    from sklearn.metrics import accuracy_score

    y_pred = model.predict_classes(X_test)
    accuracy_score(y_test, y_pred)

    # checking pred
    pred = model.predict(X_test)
    pred[5]

