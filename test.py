import os
import time
import shutil

import numpy as np
import rawpy
import glob
    
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from EDSR_model import EDSR
from model import SeeInDark


def select_short(input_dir, gt_dir, path):
    # get train and test IDs
    train_fns = glob.glob(gt_dir + '0*.ARW')
    train_ids = []
    for i in range(len(train_fns)):
        _, train_fn = os.path.split(train_fns[i])
        train_ids.append(int(train_fn[0:5]))

    test_fns = glob.glob(gt_dir + '/1*.ARW')
    test_ids = []
    for i in range(len(test_fns)):
        _, test_fn = os.path.split(test_fns[i])
        test_ids.append(int(test_fn[0:5]))

    path_list = []
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        # in_path = in_files[np.random.randint(0, len(in_files))]
        for in_path in in_files:
            print(in_path)
            path_list.append(in_path)

    with open(path, 'w') as file:
        for path in path_list:
            print(path)
            file.write(path + '\n')
    print('done!')


def extract_short(txt_path, short_folder):
    # 读取文件路径的文本文件
    with open(txt_path, "r") as file:
        file_contents = file.readlines()

    # 关闭文件
    file.close()

    # 创建目标文件夹
    if not os.path.exists(short_folder):
        os.mkdir(short_folder)

    # 复制文件到目标文件夹
    for path in file_contents:
        source_path = path.strip()  # 去除行末尾的换行符
        file_name = os.path.basename(source_path)  # 获取文件名
        destination_path = os.path.join(short_folder, file_name)  # 目标文件夹中的路径
        # shutil.copyfile(source_path, destination_path)

    print("文件已复制到目标文件夹。")


if __name__ == '__main__':
    input_dir = 'D:/Sony/Sony/short/'
    gt_dir = 'D:/Sony/Sony/long/'
    extract_short('D:\Chrome_download\Pytorch_Unet_Sony-main\short_path.txt', 'D:\Sony\short_trian')
