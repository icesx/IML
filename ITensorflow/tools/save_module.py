# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
import tensorflow as tf
from tensorflow import keras

def save_module(module,export_dir):
    print("save module ",module,"to dir ",export_dir,"..")
    tf.saved_model.save(module,export_dir=export_dir)
    print("saved")


def load_module(export_dir):
    print("load module from ",export_dir)
    return keras.models.load_model(export_dir)
