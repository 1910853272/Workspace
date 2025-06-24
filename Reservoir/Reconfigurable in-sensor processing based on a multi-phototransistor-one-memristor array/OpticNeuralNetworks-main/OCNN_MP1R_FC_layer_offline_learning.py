import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report


# Memristor量化函数
def memristor_quantize(x, levels=16):
    """
    对输入张量x进行Memristor量化处理，将值限制在0到1之间，并量化为指定的级别数。

    参数:
    - x: 输入的PyTorch张量。
    - levels: 量化的级别数，默认为16（即4位量化）。

    返回:
    - x_quantized: 量化后的张量。
    """
    # 将x的值限制在0到1之间
    x_clipped = torch.clamp(x, 0, 1)
    # 将x_clipped乘以(levels-1)，四舍五入后再除以(levels-1)进行量化
    x_quantized = torch.round(x_clipped * (levels - 1)) / (levels - 1)
    return x_quantized


# 神经网络模型定义
class SimpleNet(nn.Module):
    """
    一个简单的全连接神经网络模型，包含两个线性层和一个ReLU激活函数。
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 第一个全连接层，将输入特征从36维映射到20维
        self.fc1 = nn.Linear(36, 20)
        # 第二个全连接层，将20维映射到10个输出类别
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        """
        前向传播函数，定义数据如何通过网络流动。

        参数:
        - x: 输入张量。

        返回:
        - x: 网络的输出张量。
        """
        # 通过第一个全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 通过第二个全连接层，得到最终输出
        x = self.fc2(x)
        return x


# 训练模型的函数
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型一个epoch。

    参数:
    - model: 要训练的神经网络模型。
    - device: 设备（CPU或GPU）。
    - train_loader: 训练数据的DataLoader。
    - optimizer: 优化器，用于更新模型参数。
    - epoch: 当前的训练轮数。
    """
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和目标标签移动到指定设备
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播
        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 每100个batch打印一次训练信息
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')


# 量化模型权重的函数
def quantize_model_weights(model, levels=16):
    """
    对模型的所有参数进行Memristor量化处理。

    参数:
    - model: 要量化的神经网络模型。
    - levels: 量化的级别数，默认为16。
    """
    for param in model.parameters():
        # 对每个参数张量进行量化
        param.data = memristor_quantize(param.data, levels)


# 从CSV文件加载数据的函数
def load_data_from_csv(file_path):
    """
    从CSV文件加载数据。

    参数:
    - file_path: CSV文件的路径。

    返回:
    - X: 特征数据，numpy数组。
    - y: 标签数据，numpy数组。
    """
    data = pd.read_csv(file_path)  # 读取CSV文件
    X = data.iloc[:, :-1].values  # 提取特征（所有列，除了最后一列）
    y = data.iloc[:, -1].values  # 提取标签（最后一列）
    return X, y


# 主函数
def main():
    """
    主函数，执行数据加载、预处理、模型训练、量化和评估的整个流程。
    """
    # 设置设备，如果有GPU则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载和预处理数据
    file_path = 'data.csv'  # 数据CSV文件的路径
    X, y = load_data_from_csv(file_path)  # 从CSV文件加载特征和标签

    # 标准化特征数据，使其均值为0，标准差为1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 对标签进行One-Hot编码
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))  # 训练集标签编码
    # 注意：此处假设标签是数值型，如果是字符串或其他类型，需要先进行编码

    # 将数据集分割为训练集和测试集，测试集占20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将numpy数组转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)  # 转换为类别索引
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)  # 转换为类别索引

    # 创建训练集和测试集的数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 初始化模型和优化器
    model = SimpleNet().to(device)  # 实例化SimpleNet模型，并移动到指定设备
    optimizer = optim.Adam(model.parameters())  # 使用Adam优化器

    # 训练模型，进行10个epoch
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)

    # 对模型权重进行量化
    quantize_model_weights(model, levels=16)
    print("Model weights have been quantized.")  # 打印量化完成的信息

    # 评估模型在测试集上的性能
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 在评估过程中不计算梯度，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到指定设备
            output = model(data)  # 前向传播
            # 计算交叉熵损失，reduction='sum'表示对所有样本求和
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            # 获取预测类别
            pred = output.argmax(dim=1, keepdim=True)
            # 统计正确预测的数量
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均损失和准确率
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # 打印测试集上的损失和准确率
    print(
        f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy * 100:.2f}%)')

    # 生成分类报告
    y_pred = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)  # 将数据移动到指定设备
            output = model(data)  # 前向传播
            y_pred.extend(output.argmax(dim=1).cpu().numpy())  # 获取预测类别，并移动到CPU

    y_true = y_test.cpu().numpy()  # 获取真实类别，并移动到CPU
    print(classification_report(y_true, y_pred))  # 打印分类报告


# 如果当前脚本是主程序，则执行main函数
if __name__ == '__main__':
    main()
