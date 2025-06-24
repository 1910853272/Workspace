import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
# torch.manual_seed(1)  # reproducible
EPOCH = 1  # 训练整批数据次数，训练次数越多，精度越高，为了演示，我们训练5次
BATCH_SIZE = 50  # 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = True  # 如果你已经下载好了EMNIST数据就设置 False

# EMNIST 手写字母 训练集
train_data = torchvision.datasets.EMNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
    split = 'letters'
)
# EMNIST 手写字母 测试集
test_data = torchvision.datasets.EMNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    split = 'letters'
)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 每一步 loader 释放50个数据用来学习
# 为了演示, 我们测试时提取2000个数据先
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]
#test_x = test_x.cuda() # 若有cuda环境，取消注释
#test_y = test_y.cuda() # 若有cuda环境，取消注释