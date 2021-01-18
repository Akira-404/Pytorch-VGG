import os
import numpy as np
from PIL import Image
import sys
import torch
from model.model_vgg16 import VGG
import torch.nn as nn
import torch.optim as optim

data_set = []
label_set = {"cat": 0, "dog": 1}


def get_data_set(data_dir_path):

    # 获取项目根目录
    ABSPATH = os.path.abspath(sys.argv[-1])
    ABSPATH = os.path.dirname(ABSPATH)
    print("当前项目根目录:{}".format(ABSPATH))

    # 获取训练集路径
    train_set_path = os.path.join(ABSPATH, data_dir_path)
    if not os.path.exists(train_set_path):
        print("没有训练集")

    print("训练集路径:{}".format(train_set_path))

    # 读取训练集下所有文件
    imgs = os.listdir(train_set_path)
    print("有{}张图片".format(len(imgs)))
    train_Set = np.empty((len(imgs), 3, 224, 224), dtype="float32")
    train_set = np.empty((len(imgs), 224, 224, 3), dtype="float32")
    train_label = np.empty((len(imgs)), dtype="int")

    for i in range(len(imgs)):
        img = imgs[i]
        str = img.split('.')
        img_path = os.path.join(train_set_path, img)

        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        train_set[i, :, :, :] = img
        train_label[i] = int(label_set[str[0]])

        data_set.append((img_path, int(label_set[str[0]])))
    for i in range(3):
        train_Set[:, i, :, :] = train_set[:, :, :, i]
    print("数据加载完成\n")
    return train_Set, train_label, len(imgs)


def train(train_set, train_label, all_data_count):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    batch_size = 16
    lr = 0.0005
    epoch_size = 10
    iter = all_data_count // batch_size

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        vgg = VGG(2).to(device)
    else:
        vgg = VGG(2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg.parameters(), lr=lr)

    train_loss = []
    print("开始训练\n")
    for epoch in range(epoch_size):
        print("第{}轮".format(epoch))
        for i in range(iter):
            # 每次取出一个batch size的数据进行训练
            x = train_set[i * batch_size:i * batch_size + batch_size]
            y = train_label[i * batch_size:i * batch_size + batch_size]

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.long().cuda()

            else:
                y = y.long()

            try:
                out = vgg(x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                print("epoch:{} all batch size:{} mean:{}".format(
                    epoch, i * batch_size, np.mean(train_loss)))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            # 计算误差

            train_loss = []

    # 测试
    print("训练完成\n")
    print("开始测试\n")
    total_correct = 0
    for i in range(1):
        x = train_set[i].reshape(1, 3, 224, 224)
        y = train_label[i]
        x = torch.from_numpy(x)

        if torch.cuda.is_available():
            x = x.cuda()
            out = vgg(x).gpu()
        else:
            out = vgg(x)

        out = out.detach().numpy()
        pred = np.argmax(out)
        if pred == y:
            total_correct += 1
        print(total_correct)
    acc = total_correct / all_data_count
    print("test acc:{0.3%}".format(acc))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_set, train_label, len = get_data_set("valid-set")
    train(train_set, train_label, len)
    # print(data_set)
    # print(train_set[16:32])
    # print(train_label)
