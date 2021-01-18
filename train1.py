import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from tools import DataSet

from tools.DataSet import AnimalDataSet
from model.model_vgg16 import VGG16
from model.model_vgg11 import VGG11


def train(train_set_name, valid_set_name, test_set_name):

    interval = 10
    epoch_size = 100
    batch_size = 16
    lr = 0.01

    # 获取项目根目录
    ABSPATH = os.path.abspath(__file__)
    ABSPATH = os.path.dirname(ABSPATH)

    train_set_path = os.path.join(ABSPATH, train_set_name)
    valid_set_path = os.path.join(ABSPATH, valid_set_name)
    test_set_path = os.path.join(ABSPATH, test_set_name)

    print(train_set_path)

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_tansform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    valid_tansform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_data = AnimalDataSet(train_set_path, train_tansform)
    valid_data = AnimalDataSet(valid_set_path, valid_tansform)
    test_data = AnimalDataSet(test_set_path, valid_tansform)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # net = VGG16(num_classes=2).to(device)
    net = VGG11(num_classes=2).to(device)

    net.initialize_weights()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    train_curve = list()
    valid_curve = list()

    iter_count = 0

    for epoch in range(2):

        loss_mean = 0
        correct = 0
        total = 0

        net.train().cuda()
        # net.train()

        for i, data in enumerate(train_loader):
            iter_count += 1

            inputs, labels = data
            # print("inputs:{},labels:{}".format(inputs.size(),labels))

            inputs = inputs.to(device)
            labels = labels.to(device)

            outpus = net(inputs)

            optimizer.zero_grad()

            loss = criterion(outpus, labels)

            loss.backward()

            optimizer.step()

            _, pred = torch.max(outpus.data, 1)
            total += labels.size(0)
            correct += (pred == labels).squeeze().sum().numpy()

            loss_mean += loss.item()
            train_curve.append(loss.item())

            # 每10轮打印一次训练信息
            if (i + 1) % interval == 0:
                loss_mean = loss_mean / interval
                print(
                    "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch,
                        epoch_size,
                        i +
                        1,
                        len(train_loader),
                        loss_mean,
                        correct /
                        total))
                loss_mean = 0.

        scheduler.step()
        if (epoch + 1) % 1 == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted ==
                                    labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val / valid_loader.__len__())
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch,
                        epoch_size,
                        j +
                        1,
                        len(valid_loader),
                        loss_val,
                        correct_val /
                        total_val))
    # 打印图像信息
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * interval
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

    for i, data in enumerate(test_loader):
        # 前向传播
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        a = 1 if predicted.numpy()[0] == 0 else 100
        print("预测是{}".format(a))


if __name__ == "__main__":
    train_set_name = "train-set"
    valid_set_name = "valid-set"
    test_set_name = "valid-set"
    train(train_set_name, valid_set_name, test_set_name)
