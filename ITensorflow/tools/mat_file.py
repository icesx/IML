# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
# 2/1/21
import scipy.io as sio
import os
import shutil
from pathlib import Path


def mat_dict(file):
    mat_data = sio.loadmat(file)
    print(mat_data.values())
    print(mat_data.keys())
    lables = list(mat_data.keys())[-1]
    print("lable is %s" % lables)
    data = mat_data[lables]
    index = 0
    dict = {}
    for i in data[0, :]:
        index += 1
        file = "image_{:0>5d}.jpg".format(index)
        dict[file] = i
    return dict


def mv_dict(dict, workdir):
    for key in dict:
        folder = dict[key]
        workdir_folder = workdir + "/" + str(folder)
        if os.path.exists(workdir_folder) is False:
            os.mkdir(workdir_folder)
        workdir_key = workdir + "/" + key
        folder_key = workdir_folder + "/" + key
        print("mv from %s, to %s" % (workdir_key, folder_key))
        shutil.move(workdir_key, folder_key)


if __name__ == '__main__':
    dict = mat_dict("/WORK/datasset/102flowers/imagelabels.mat")
    mv_dict(dict, "/WORK/datasset/102flowers/jpg")
