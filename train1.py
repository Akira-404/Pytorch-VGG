import torch
from tools import DataSet


def train(train_set_name):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')


if __name__ == "__main__":
    train_set_path = "traiin-set"

    dogdataset = DataSet(train_set_path, "dog")
    catdataset = DataSet(train_set_path, "cat")
