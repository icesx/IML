# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/2/21
import os

import matplotlib.pyplot as plt
from tensorflow import keras

from tools.load_data import keras_load_data


def save_fashion_mnist(workdir="/WORK/datasset/fashion_mnist"):
    class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.fashion_mnist)
    __save_dateset(class_names, train_images, train_labels, workdir)


def __save_dateset(class_names, train_images, train_labels, workdir):
    for index, (image, lable) in enumerate(zip(train_images, train_labels)):
        workdir_folder = workdir + "/" + class_names[lable]
        if os.path.exists(workdir_folder) is False:
            os.mkdir(workdir_folder)
        plt.imsave(workdir_folder + "/" + str(index) + ".jpg", image)


def save_mnist(workdir="/WORK/datasset/mnist"):
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.mnist)
    class_names = ['0', '1', "2", "3", "4", "5", "6", "7", "8", "9"]
    __save_dateset(class_names, train_images, train_labels, workdir)


if __name__ == '__main__':
    save_mnist()
