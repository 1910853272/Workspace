import torch.nn
import torchvision  # 该包包含了与计算机视觉有关的内容，包括常见的数据集和数据转换。
import numpy as np
import pandas as pd
import math
from torch.utils.data import DataLoader  # 从 PyTorch 中导入 DataLoader 来处理数据的批处理和随机化。
from MLP_model import MLP  # 导入已经构建好的 MLP 模型。
# from ANN_model import ANN
# from CNN_model import CNN
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from curve import Y  # 从外部文件 curve.py 导入 LTP 和 LTD 曲线的电导值。

# 设置随机种子，保证结果可重复
torch.manual_seed(128)

# 参数设定
Wmax = 1  # 权重的最大值（表示突触强度的范围）
lr = 0.2  # 学习率，用于更新权重

# 二分查找函数，用于在数组中查找与目标值最接近的元素的索引
def search(arr, e):
    # 初始化变量：low 表示查找范围的下界，high 表示上界，idx 用于存储找到的元素索引，初始为 -1 表示未找到
    low = 0
    high = len(arr) - 1
    idx = -1
    
    # 循环进行二分查找，直到 low 大于 high 时停止
    while low <= high:
        # 计算中间位置的索引 mid，向下取整
        mid = int((low + high) / 2)
        
        # 如果目标值 e 等于中间值 arr[mid] 或者 low 与 mid 相等（表示搜索范围已经最小化），则找到了最接近的索引
        if e == arr[mid] or mid == low:
            idx = mid  # 将 mid 设置为找到的索引
            break  # 结束查找
        # 如果目标值 e 大于中间值，说明目标值在右半部分，更新 low 为 mid
        elif e > arr[mid]:
            low = mid
        # 如果目标值 e 小于中间值，说明目标值在左半部分，更新 high 为 mid
        elif e < arr[mid]:
            high = mid
    
    # 检查找到的索引是否是最佳匹配
    # 如果找到的 idx 不是最后一个元素，并且目标值 e 与 arr[idx+1] 的差值比 e 与 arr[idx] 的差值更小
    # 说明更靠近 arr[idx+1]，则将 idx 加 1
    if idx + 1 < len(arr) and abs(e - arr[idx]) > abs(e - arr[idx + 1]):
        idx += 1
    
    # 返回找到的最接近的索引
    return idx


# 第一步：根据 LTP（长期增强）和 LTD（长期减弱）曲线准备电导值
cond = []
for num in Y:
    cond.append(num)

Ptot = int((len(cond)) / 2)  # 将电导值分为两部分，分别对应 LTP 和 LTD

# 将电导值映射到权重大小范围
cond_max = max(cond)  # 最大电导值
cond_min = min(cond)  # 最小电导值
for i, k in enumerate(cond):
    cond[i] = k - ((cond_max + cond_min) / 2)  # 将电导值通过最大值和最小值的平均值进行偏移

# 将电导值缩放至权重范围
A = Wmax / max(cond)  # 权重的缩放因子，基于电导最大值
for i, k in enumerate(cond):
    cond[i] = k * A  # 对电导值进行缩放

# 将电导值分为 LTP 和 LTD 部分
LTP = cond[0:Ptot]
LTP.sort()  # 对 LTP 值排序以便于后续查找
LTD = cond[Ptot:Ptot * 2]
LTD.sort()  # 对 LTD 值排序以便于后续查找

# 第二步：加载 MNIST 数据集进行训练和测试
train_data = torchvision.datasets.MNIST(
    root="data",  # 将 MNIST 数据保存在 "data" 文件夹中
    download=True,  # 如果数据集未下载过，则从网络下载
    train=True,  # 加载训练数据集
    transform=torchvision.transforms.ToTensor()  # 将图像数据转换为 Tensor 格式
)

test_data = torchvision.datasets.MNIST(
    root="data",  # 将 MNIST 数据保存在 "data" 文件夹中
    download=True,  # 如果数据集未下载过，则从网络下载
    train=False,  # 加载测试数据集
    transform=torchvision.transforms.ToTensor()  # 将图像数据转换为 Tensor 格式
)

# 第三步：创建 DataLoader 以进行批量处理训练和测试数据集
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)  # 将训练数据按 128 批次打乱后加载
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)  # 测试数据按 128 批次加载，不进行打乱

# 第四步：初始化模型并设置权重
model = MLP()  # 实例化 MLP 模型
# model = ANN()  # 实例化 MLP 模型
# model = CNN()  # 实例化 MLP 模型
dict = model.state_dict()  # 获取模型的状态字典（权重、偏置等）
weight = dict['net.1.weight']  # 访问第二层的权重（在 0 基系统中是第 1 层）
torch.nn.init.normal_(weight, mean=0.0, std=0.03)  # 使用正态分布（均值=0，标准差=0.03）初始化权重

# 第五步：设置损失函数和优化器
loss_func = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于分类任务
optimizer = torch.optim.SGD(model.parameters(), lr=0.007)  # 使用随机梯度下降优化器，学习率为 0.007

# 第六步：设置训练的迭代次数
nums_epochs = 50  # 训练 50 个循环（epochs）
accuracy_list = []  # 用于记录准确率的列表
epoch_list = []  # 用于记录时间或训练轮次的列表

# 第七步：开始训练模型
flag_ltp = 0
flag_ltd = 0

for cnt in range(nums_epochs):  # 遍历每个 epoch
    for batch, (imgs, labels) in enumerate(train_dataloader):  # 遍历每个批次的数据
        correct = 0  # 初始化正确预测的数量
        outputs = model(imgs)  # 前向传播：计算模型输出
        loss = loss_func(outputs, labels.flatten().long())  # 计算损失（交叉熵）
        optimizer.zero_grad()  # 清空之前计算的梯度
        loss.backward()  # 反向传播：计算参数的梯度

        # 基于计算的梯度更新权重
        for param in model.parameters():  # 遍历模型中的每个参数（权重和偏置）
            weight = param.data
            grad = param.grad
            weight_num = weight.data.numpy()  # 将权重转换为 NumPy 数组
            grad_num = grad.data.numpy()  # 将梯度转换为 NumPy 数组
            
            # 遍历梯度，基于电导和学习规则调整权重
            row, col = grad_num.shape  # 获取梯度矩阵的形状
            for i in range(row):
                for j in range(col):
                    Pulse = int(abs(grad_num[i, j] * lr / (Wmax / Ptot / 2)))  # 计算更新权重的脉冲数
                    if grad_num[i, j] < 0:  # 长期增强（LTP）情况
                        flag_ltp += 1
                        p_LTP = search(LTP, weight_num[i, j])  # 查找最接近的 LTP 电导值
                        if p_LTP + Pulse >= Ptot:
                            weight_num[i, j] = LTP[Ptot - 1]  # 限制最大 LTP 值
                        else:
                            weight_num[i, j] = LTP[p_LTP + Pulse]  # 使用 LTP 更新权重
                    elif grad_num[i, j] > 0:  # 长期减弱（LTD）情况
                        flag_ltd += 1
                        p_LTD = search(LTD, weight_num[i, j])  # 查找最接近的 LTD 电导值
                        if p_LTD - Pulse <= 0:
                            weight_num[i, j] = LTD[0]  # 限制最小 LTD 值
                        else:
                            weight_num[i, j] = LTD[p_LTD - Pulse]  # 使用 LTD 更新权重
                    else:
                        continue  # 如果梯度为零，则不更新权重
            weight.data = torch.Tensor(weight_num)  # 更新模型的权重

    # 第八步：每个 epoch 结束后使用测试数据集评估模型性能
    total_loss = 0
    with torch.no_grad():  # 禁用梯度计算以进行评估
        correct = 0  # 重置正确预测计数
        for imgs, labels in test_dataloader:  # 遍历测试数据集
            outputs = model(imgs)  # 前向传播：获取模型输出
            predicted = torch.max(outputs.data, 1)[1]  # 获取预测类别
            correct += (predicted == labels.flatten()).sum()  # 计算正确预测的数量

        # 计算准确率
        acc = float(correct * 100) / 10000  # 计算准确率百分比
        accuracy_list.append(acc)  # 保存当前 epoch 的准确率
        epoch_list.append(cnt + 1)  # 保存当前的 epoch 轮数
        print(f"Epoch {cnt + 1}, Accuracy: {acc:.2f}%")  # 输出当前 epoch 的准确率

# 训练结束后，您可以选择保存模型和权重
# torch.save(model, "my_mnistmlp.nn")
# print('模型和参数已保存。')

# 绘制准确率随训练周期变化的折线图
plt.figure(figsize=(10, 6))  # 设置图形的大小
plt.plot(epoch_list, accuracy_list, label='Accuracy')  # 绘制横轴为训练周期，纵轴为准确率的折线图
plt.xlabel('Epoch')  # 为横轴添加标签 'Epoch'，表示训练的轮数
plt.ylabel('Accuracy (%)')  # 为纵轴添加标签 'Accuracy (%)'，表示分类准确率的百分比
plt.title('Model Accuracy Over Epochs')  # 为图形添加标题 'Model Accuracy Over Epochs'，即"模型准确率随训练周期变化"
plt.grid(True)  # 启用网格线，使图形更容易阅读
plt.legend()  # 添加图例，用于标记曲线含义
plt.show()  # 显示绘制好的图形
