# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
from deployer.deploy import Deploy


def main():
    print("please input the conrect param as : python deploy.py root@xx.xx.xx.xx:/temp /home/user/Downloads/log4py-1.3.zip log4py-1.3_xxxx.zip")
    remote_dir = "/home/bjrdc/code/"
    local_dir = "/ICESX/workspaceTensorflow/ITensorflow"
    Deploy(local_dir,host="bjrdc5",remote_dir=remote_dir,user="bjrdc",
           password="zgjx@321").zip().scp_file().ssh_unzip()


if __name__ == '__main__':
    main()
