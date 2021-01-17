import os
from PIL import Image
import random
import sys
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, data_dir_name, data_name, transfrom=None):
        self.data_set = self.get_data(data_dir_name,data_name)
        self.transfrom = transfrom

    def __getitem__(self, index):
        img_path, label = self.data_set[index]
        img = Image.open(img_path).convert('RGB')

        if self.transfrom is not None:
            img = self.transfrom(img)
        return img, label

    def __len__(self):
        return len(self.data_set)

    # 输入数据集的名字
    def get_data(self, data_dir_name, data_name):

        data_set = []

        # 获取项目根目录
        ABSPATH = os.path.abspath(sys.argv[-1])
        ABSPATH = os.path.dirname(ABSPATH)
        print("当前项目根目录:{}".format(ABSPATH))

        # 获取训练集路径
        train_set_path = os.path.join(ABSPATH, data_dir_name)
        if not os.path.exists(train_set_path):
            print("没有训练集")
            return
        print("训练集路径:{}".format(train_set_path))

        # 读取训练集下所有文件
        imgs = os.listdir(train_set_path)
        imgs = list(filter(lambda x: x.startswith(data_name), imgs))
        print("有{}张{}的图片".format(len(imgs), data_name))

        for i in range(len(imgs)):
            img = imgs[i]
            img_path = os.path.join(train_set_path, img)
            data_set.append((img_path, 1))
        return data_set
