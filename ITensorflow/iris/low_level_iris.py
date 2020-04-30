# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum
from sklearn.datasets import load_iris
from typing import Callable, Iterable, List, Tuple


class HyperParams(Enum):
    ACTIVATION = tf.nn.relu
    BATCH_SIZE = 5
    EPOCHS = 500
    HIDDEN_NEURONS = 10
    NORMALIZER = tf.nn.softmax
    OUTPUT_NEURONS = 3
    OPTIMIZER = tf.keras.optimizers.Adam


iris = load_iris()
xdat = iris.data
ydat = iris.target


class Data:

    def __init__(self, xdat: np.ndarray, ydat: np.ndarray, ratio: float = 0.3) -> Tuple:
        self.xdat = xdat
        self.ydat = ydat
        self.ratio = ratio

    def partition(self) -> None:
        scnt = self.xdat.shape[0] / np.unique(self.ydat).shape[0]
        ntst = int(self.xdat.shape[0] * self.ratio / (np.unique(self.ydat)).shape[0])
        idx = np.random.choice(np.arange(0, self.ydat.shape[0] / np.unique(self.ydat).shape[0], dtype=int), ntst,
                               replace=False)
        for i in np.arange(1, np.unique(self.ydat).shape[0]):
            idx = np.concatenate(
                (idx, np.random.choice(np.arange((scnt * i), scnt * (i + 1), dtype=int), ntst, replace=False)))

        self.xtrn = self.xdat[np.where(~np.in1d(np.arange(0, self.ydat.shape[0]), idx))[0], :]
        self.ytrn = self.ydat[np.where(~np.in1d(np.arange(0, self.ydat.shape[0]), idx))[0]]
        self.xtst = self.xdat[idx, :]
        self.ytst = self.ydat[idx]

    def to_tensor(self, depth: int = 3) -> None:
        self.xtrn = tf.convert_to_tensor(self.xtrn, dtype=np.float32)
        self.xtst = tf.convert_to_tensor(self.xtst, dtype=np.float32)
        self.ytrn = tf.convert_to_tensor(tf.one_hot(self.ytrn, depth=depth))
        self.ytst = tf.convert_to_tensor(tf.one_hot(self.ytst, depth=depth))

    def batch(self, num: int = 16) -> None:
        try:
            size = self.xtrn.shape[0] / num
            if self.xtrn.shape[0] % num != 0:
                sizes = [tf.floor(size).numpy().astype(int) for i in range(num)] + [self.xtrn.shape[0] % num]
            else:
                sizes = [tf.floor(size).numpy().astype(int) for i in range(num)]

            self.xtrn_batches = tf.split(self.xtrn, num_or_size_splits=sizes, axis=0)
            self.ytrn_batches = tf.split(self.ytrn, num_or_size_splits=sizes, axis=0)

            num = int(self.xtst.shape[0] / sizes[0])
            if self.xtst.shape[0] % sizes[0] != 0:
                sizes = [sizes[i] for i in range(num)] + [self.xtst.shape[0] % sizes[0]]
            else:
                sizes = [sizes[i] for i in range(num)]

            self.xtst_batches = tf.split(self.xtst, num_or_size_splits=sizes, axis=0)
            self.ytst_batches = tf.split(self.ytst, num_or_size_splits=sizes, axis=0)
        except:
            self.xtrn_batches = [self.xtrn]
            self.ytrn_batches = [self.ytrn]
            self.xtst_batches = [self.xtst]
            self.ytst_batches = [self.ytst]


data = Data(xdat, ydat)
data.partition()
data.to_tensor()
data.batch(HyperParams.BATCH_SIZE.value)


class Dense:

    def __init__(self, i: int, o: int, f: Callable[[tf.Tensor], tf.Tensor],
                 initializer: Callable = tf.random.normal) -> None:
        self.w = tf.Variable(initializer([i, o]))
        self.b = tf.Variable(initializer([o]))
        self.f = f

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if callable(self.f):
            return self.f(tf.add(tf.matmul(x, self.w), self.b))
        else:
            return tf.add(tf.matmul(x, self.w), self.b)


class Chain:

    def __init__(self, layers: List[Iterable[Dense]]) -> None:
        self.layers = layers

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        self.out = x;
        self.params = []
        for l in self.layers:
            self.out = l(self.out)
            self.params.append([l.w, l.b])

        self.params = [j for i in self.params for j in i]
        return self.out

    def backward(self, inputs: tf.Tensor, targets: tf.Tensor) -> None:
        grads = self.grad(inputs, targets)
        self.optimize(grads, 0.001)

    def loss(self, preds: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                targets, preds
            )
        )

    def grad(self, inputs: tf.Tensor, targets: tf.Tensor) -> List:
        with tf.GradientTape() as g:
            error = self.loss(self(inputs), targets)

        return g.gradient(error, self.params)

    def optimize(self, grads: List[tf.Tensor], rate: float) -> None:
        opt = HyperParams.OPTIMIZER.value(learning_rate=rate)
        opt.apply_gradients(zip(grads, self.params))


model = Chain([
    Dense(data.xtrn.shape[1], HyperParams.HIDDEN_NEURONS.value, HyperParams.ACTIVATION),
    Dense(HyperParams.HIDDEN_NEURONS.value, HyperParams.OUTPUT_NEURONS.value, HyperParams.NORMALIZER)
])


def accuracy(y, yhat):
    j = 0;
    correct = []
    for i in tf.argmax(y, 1):
        if i == tf.argmax(yhat[j]):
            correct.append(1)

        j += 1

    num = tf.cast(tf.reduce_sum(correct), dtype=tf.float32)
    den = tf.cast(y.shape[0], dtype=tf.float32)
    return num / den


epoch_trn_loss = []
epoch_tst_loss = []
epoch_trn_accy = []
epoch_tst_accy = []
for j in range(HyperParams.EPOCHS.value):
    trn_loss = [];
    trn_accy = []
    for i in range(len(data.xtrn_batches)):
        model.backward(data.xtrn_batches[i], data.ytrn_batches[i])
        ypred = model(data.xtrn_batches[i])
        trn_loss.append(model.loss(ypred, data.ytrn_batches[i]))
        trn_accy.append(accuracy(data.ytrn_batches[i], ypred))

    trn_err = tf.reduce_mean(trn_loss).numpy()
    trn_acy = tf.reduce_mean(trn_accy).numpy()

    tst_loss = [];
    tst_accy = []
    for i in range(len(data.xtst_batches)):
        ypred = model(data.xtst_batches[i])
        tst_loss.append(model.loss(ypred, data.ytst_batches[i]))
        tst_accy.append(accuracy(data.ytst_batches[i], ypred))

    tst_err = tf.reduce_mean(tst_loss).numpy()
    tst_acy = tf.reduce_mean(tst_accy).numpy()

    epoch_trn_loss.append(trn_err)
    epoch_tst_loss.append(tst_err)
    epoch_trn_accy.append(trn_acy)
    epoch_tst_accy.append(tst_acy)

    if j % 20 == 0:
        print(
            "Epoch: {0:4d} \t Training Error: {1:.4f} \t Testing Error: {2:.4f} \t Accuracy Training: {3:.4f} \t Accuracy Testing: {4:.4f}".format(
                j, trn_err, tst_err, trn_acy, tst_acy))

df = pd.DataFrame({
    "trn_loss": epoch_trn_loss,
    "trn_accy": epoch_trn_accy,
    "tst_loss": epoch_tst_loss,
    "tst_accy": epoch_tst_accy
})

df.to_csv("../tf2_output_normal_initializer_batch_size_" + str(HyperParams.BATCH_SIZE.value) + ".csv")
