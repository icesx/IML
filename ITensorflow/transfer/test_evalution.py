# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from transfer.data_loader import DataLoader
from transfer.import_global import *

if __name__ == '__main__':
    data_loader = DataLoader(IMG_SIZE, BATCH_SIZE)
    steps_per_epoch = round(data_loader.num_train_examples) // BATCH_SIZE
    validation_steps = 20

    loss1, accuracy1 = vgg16.evaluate(data_loader.validation_batches, steps=20)
    loss2, accuracy2 = googlenet.evaluate(data_loader.validation_batches, steps=20)
    loss3, accuracy3 = resnet.evaluate(data_loader.validation_batches, steps=20)

    print("--------VGG16---------")
    print("Initial loss: {:.2f}".format(loss1))
    print("Initial accuracy: {:.2f}".format(accuracy1))
    print("---------------------------")

    print("--------GoogLeNet---------")
    print("Initial loss: {:.2f}".format(loss2))
    print("Initial accuracy: {:.2f}".format(accuracy2))
    print("---------------------------")

    print("--------ResNet---------")
    print("Initial loss: {:.2f}".format(loss3))
    print("Initial accuracy: {:.2f}".format(accuracy3))
    print("---------------------------")
