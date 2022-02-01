# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_lables),(test_images,test_lables) = fashion_mnist.load_data()
class_names = class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                             'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print(train_images.shape)
len(train_lables)
print(train_lables)
print(test_images.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
#set the image to normal
train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_lables[i]])
plt.show()
# set the nn
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128,activation=tf.nn.relu),
                          keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_images,train_lables,epochs=150)
test_loss,test_acc = model.evaluate(test_images,test_lables)
print('Test accuracy',test_acc)
# prediction
predictions = model.predict(test_images)
print(predictions[0])
print(np.max(predictions[0]))
