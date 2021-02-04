# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/2/21
import os

import matplotlib.pyplot as plt
import numpy
from tensorflow import keras

from tools.load_data import keras_load_data


def save_fashion_mnist(workdir="/WORK/datasset/fashion_mnist"):
    class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.fashion_mnist)
    __save_dateset(class_names, train_images, train_labels, workdir + "/train")
    __save_dateset(class_names, test_images, test_labels, workdir + "/val")


def __save_dateset(class_names, train_images, train_labels, workdir):
    for index, (image, lable) in enumerate(zip(train_images, train_labels)):
        workdir_folder = workdir + "/" + class_names[lable]
        if os.path.exists(workdir_folder) is False:
            os.mkdir(workdir_folder)
        plt.imsave(workdir_folder + "/" + str(index) + ".jpg", image)


def save_mnist(workdir="/WORK/datasset/mnist"):
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.mnist)
    class_names = ['0', '1', "2", "3", "4", "5", "6", "7", "8", "9"]
    __save_dateset(class_names, train_images, train_labels, workdir + "/train")
    __save_dateset(class_names, test_images, test_labels, workdir + "/val")


def save_cifar_10(workdir="/WORK/datasset/cifar10"):
    train_images, test_images, train_labels, test_labels = keras_load_data(keras.datasets.cifar10)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    __save_dateset(class_names, train_images, numpy.array(train_labels).flatten().tolist(), workdir + "/train")
    __save_dateset(class_names, test_images, numpy.array(test_labels).flatten().tolist(), workdir + "/val")


if __name__ == '__main__':
    save_cifar_10()
