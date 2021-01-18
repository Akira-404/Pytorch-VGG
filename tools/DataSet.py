import os
from PIL import Image
import sys
from torch.utils.data import Dataset

label = {"cat": 0, "dog": 1}


class AnimalDataSet(Dataset):

    def __init__(self, train_set_path, transfrom=None):
        self.data_set = self.get_data(train_set_path)
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
    # 读取每一个图片的路径和对应的标签，组成一个元组
    def get_data(self, train_set_path):

        data_set = []

        print("数据集路径:{}".format(train_set_path))
        if not os.path.exists(train_set_path):
            print("没有数据集")
            return

        # 读取训练集下所有文件
        imgs = os.listdir(train_set_path)
        print(len(imgs))
        # imgs = list(filter(lambda x: x.startswith(data_name), imgs))
        # print("有{}张{}的图片".format(len(imgs), data_name))

        for i in range(len(imgs)):
            img = imgs[i]
            str = img.split('.')
            img_path = os.path.join(train_set_path, img)
            data_set.append((img_path, int(label[str[0]])))
        return data_set


if __name__ == "__main__":

    ds = AnimalDataSet("..\\train-set")
    set = ds.get_data("..\\train-set")
    print(set)
