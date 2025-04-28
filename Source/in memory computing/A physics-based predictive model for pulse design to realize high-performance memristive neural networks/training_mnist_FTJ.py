import torch.nn
import torchvision  # 导入所需要的包，这个包里包含了机器视觉有关的内容
import numpy as np
import pandas as pd
import math
from torch.utils.data import DataLoader  # 导入Dataloader
from mnist_model import mlp  # 导入构建好的模型,重新设置了一个mnist的模型
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
# 导入G曲线完整取值
# from curve1 import Y
from curve2 import Y
# from curve3 import Y
torch.manual_seed(128)

# 参数设定
Wmax = 1  # 权重范围
lr = 0.2

def search(arr, e):
    low = 0
    high = len(arr) - 1
    idx = -1
    while low <= high:
        mid = int((low + high) / 2)
        if e == arr[mid] or mid == low:
            idx = mid
            break
        elif e > arr[mid]:
            low = mid
        elif e < arr[mid]:
            high = mid
    if idx + 1 < len(arr) and abs(e - arr[idx]) > abs(e - arr[idx + 1]):
        idx += 1

    return idx


# def save_excel(temp, dir_):
#     # 写入excel
#     data_df = pd.DataFrame(temp)
#     writer = pd.ExcelWriter(dir_)
#     data_df.to_excel(writer, sheet_name='coding', float_format='%.8f')  # float_format 控制精度
#     writer.save()
#     writer.close()

# 这里应该就是主函数开始了
# 列出电导值，按LTP,LTD曲线
cond = []
for num in Y:
    cond.append(num)
Ptot = int((len(cond)) / 2)
# (将电导值映射到权重大小中)
cond_max = max(cond)  # 最大的电导值
cond_min = min(cond)
for i, k in enumerate(cond):
    cond[i] = k - ((cond_max + cond_min) / 2)  # 将电导值映射为规定的电导范围
A = Wmax / max(cond)  # 用权重最大除以电导最大,此时的权重最大值是由每一次决定的
for i, k in enumerate(cond):
    cond[i] = k * A

# #把G分为LTP和LTD
LTP = cond[0:Ptot]
LTP.sort()
LTD = cond[Ptot:Ptot * 2]
LTD.sort()

# 训练数据集
train_data = torchvision.datasets.MNIST(
    root="data",  # 表示把MINST保存在data文件夹下
    download=True,  # 表示需要从网络上下载。下载过一次后，下一次就不会再重复下载了
    train=True,  # 表示这是训练数据集
    transform=torchvision.transforms.ToTensor()
    # 要把数据集中的数据转换为pytorch能够使用的Tensor类型
)

# 测试数据集
test_data = torchvision.datasets.MNIST(
    root="data",  # 表示把MINST保存在data文件夹下
    download=True,  # 表示需要从网络上下载。下载过一次后，下一次就不会再重复下载了
    train=False,  # 表示这是测试数据集
    transform=torchvision.transforms.ToTensor()
    # 要把数据集中的数据转换为pytorch能够使用的Tensor类型
)

# 创建两个Dataloader, 包尺寸为64
# 训练用的Dataloader
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
# 测试用的Dataloader
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

# 实例化模型
idx = 0
model = mlp()
dict = model.state_dict()
weight = dict['net.1.weight']
# here we initialise the weight
torch.nn.init.normal_(weight,mean=0.0,std=0.03)

# 交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()

# 优化器,no nse
optimizer = torch.optim.SGD(model.parameters(), lr=0.007)

# 定义训练次数
cnt_epochs = 10  # 训练10个循环
bcc = []
time = []
# 循环训练
flag_ltp = 0
flag_ltd = 0
acc_out = 0
flag=0

weight_dict = []

for cnt in range(cnt_epochs):
    # 把训练集中的数据训练一遍
    for batch, (imgs, labels) in enumerate(train_dataloader):
        correct = 0
        outputs = model(imgs)
        loss = loss_func(outputs, labels.flatten().long())
        optimizer.zero_grad()  # 注意清空优化器的梯度，防止累计
        loss.backward()
        for j in model.parameters():
            weight = j.data
            grad = j.grad
            weight_num = weight.data.numpy()  # 将tensor转换成numpy
            grad_num = grad.data.numpy()
            # 遍历梯度
            [row, col] = grad_num.shape  # 他的权重应该是始终在电导范围内的，但是权重范围会有变化
            for i in range(row):
                for j in range(col):  # 这电导因为其自身性质不会超出最大值
                    Pulse = int(abs(grad_num[i, j] * lr / (Wmax / Ptot / 2)))  # 多脉冲输出权重
                    if grad_num[i, j] < 0:
                        flag_ltp += 1
                        p_LTP = search(LTP, weight_num[i, j])
                        if p_LTP + Pulse >= Ptot:
                            weight_num[i, j] = LTP[Ptot - 1]
                            continue
                        weight_num[i, j] = LTP[p_LTP + Pulse]
                    elif grad_num[i, j] > 0:
                        flag_ltd += 1
                        p_LTD = search(LTD, weight_num[i, j])
                        if p_LTD - Pulse <= 0:
                            weight_num[i, j] = LTD[0]
                            continue
                        weight_num[i, j] = LTD[p_LTD - Pulse]
                    else:
                        continue
            weight.data = torch.Tensor(weight_num)

        # 用测试集测试一下当前训练过的神经网络
    total_loss = 0  # 保存这次测试总的loss
    with torch.no_grad():  # 下面不需要反向传播，所以不需要自动求导
        for i, (imgs, labels) in enumerate(test_dataloader):
            outputs = model(imgs)
            # 计算准确率
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == labels.flatten()).sum()
            acc = float(correct * 100) / 10000
            x = cnt + 1
    bcc.append(acc)
    time.append(x)
    print(acc)


# 保存训练的结果（包括模型和参数）
# torch.save(model, "my_mnistmlp.nn")
# print('already save the model')