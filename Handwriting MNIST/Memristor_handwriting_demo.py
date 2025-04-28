import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import datasets, transforms
from torch.autograd import Variable

# 定义忆阻器参数
OM = -1  # 代表忆阻器的特性
IREAD = 1e-9  # 读取电流
MVT = 0.144765  # 热电压
CR = 1  # 常数
BB45 = 5.1e-5  # 衰减常数
CRPROG = 0.48  # 编程常数
VPROG = 4.5  # 编程电压
VDS = 2  # 源漏电压

# 定义Memristor层（忆阻器层）
class Memristor_layer(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, lower, upper, bias=True):
        super(Memristor_layer, self).__init__()
        self.in_features = in_features  # 输入特征的数量
        self.out_features = out_features  # 输出特征的数量
        self.lower = lower  # 权重下限
        self.upper = upper  # 权重上限
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # 权重参数
        self.Vth = torch.ones_like(self.weight.data) + 0.01 * torch.rand_like(self.weight.data)  # 阈值电压
        self.Ids = lambda v: IREAD * np.exp(CR * VDS / MVT) * torch.exp(-v / MVT)  # 电流函数
        self.dvthdt = torch.ones_like(self.weight.data)  # 电压变化率

        # 如果需要偏置，初始化偏置参数
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()  # 重置参数
        self.weight.data = (CR / MVT * self.Ids(self.Vth)) ** OM  # 初始化权重

    def reset_parameters(self):
        # 权重和偏置的随机初始化
        nn.init.uniform_(self.weight, self.lower, self.upper)
        if self.bias is not None:
            nn.init.uniform_(self.bias, self.lower, self.upper)

    def forward(self, input):
        # 前向传播，线性变换
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)

    def update_weight(self, lr):
        # 更新权重
        self.weight.data.requires_grad = False  # 禁用权重的梯度计算
        self.dvthdt = BB45 * (CRPROG * VPROG - self.Vth)  # 计算阈值电压变化率
        self.Vth = self.Vth + torch.mul(self.dvthdt, torch.sign(torch.relu(OM * self.weight.grad)))  # 更新阈值电压
        self.weight.data = (CR / MVT * self.Ids(self.Vth)) ** OM  # 更新权重
        self.weight.data.requires_grad = True  # 重新启用权重的梯度计算


# 定义完整的网络，使用卷积层和忆阻器层（MAC Crossbar层）
class CNN_with_Crossbar(nn.Module):
    def __init__(self, args):
        super(CNN_with_Crossbar, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1个通道（灰度图），输出32个通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32个通道，输出64个通道

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，2x2核，步幅为2

        # 全连接层
        self.fc1_pos = Memristor_layer(64 * 7 * 7, 128, lower=args.lower, upper=args.upper)  # 正权重的忆阻器层
        self.fc1_neg = Memristor_layer(64 * 7 * 7, 128, lower=args.lower, upper=args.upper)  # 负权重的忆阻器层
        
        self.fc2_pos = Memristor_layer(128, 10, lower=args.lower, upper=args.upper)  # 最后一层分类层，正权重
        self.fc2_neg = Memristor_layer(128, 10, lower=args.lower, upper=args.upper)  # 最后一层分类层，负权重

        self.criterion = nn.CrossEntropyLoss(reduction='sum')  # 交叉熵损失
        self.lr = args.lr  # 学习率

    def forward(self, x):
        # 前向传播过程
        x = self.pool(F.relu(self.conv1(x)))  # 第一个卷积层 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二个卷积层 -> 激活 -> 池化
        
        x = x.view(-1, 64 * 7 * 7)  # 展平，准备输入全连接层
        x = F.relu(self.fc1_pos(x) - self.fc1_neg(x))  # 计算全连接层输出
        x = self.fc2_pos(x) - self.fc2_neg(x)  # 最后输出层，计算分类结果
        
        return x

    def update_weights(self):
        # 更新忆阻器的权重
        self.fc1_pos.update_weight(self.lr)
        self.fc1_neg.update_weight(self.lr)
        self.fc2_pos.update_weight(self.lr)
        self.fc2_neg.update_weight(self.lr)

    def optimizer_step(self, epoch):
        # 调整学习率，epoch时进行
        self.lr /= (10 ** epoch)


# 参数类，用于设置学习率和权重范围
class Args:
    def __init__(self, lr=0.01, lower=-0.1, upper=0.1):
        self.lr = lr  # 学习率
        self.lower = lower  # 权重下限
        self.upper = upper  # 权重上限


# 数据处理，包括转换和标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor格式
    transforms.Normalize([0.5], [0.5])  # 标准化处理
])

# 训练集，下载MNIST数据集
train_data = datasets.MNIST(
    root='./data/', train=True, transform=transform, download=True
)

# 测试集
test_data = datasets.MNIST(
    root='./data/', train=False, transform=transform
)

# 每批装载的数据图片设置为64
batch_size = 64

# 加载训练数据集
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True  # 随机加载
)

# 加载测试数据集
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False  # 顺序加载
)

# 创建模型
model = CNN_with_Crossbar(Args())
print(model)  # 打印模型结构

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters())  # Adam优化器
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失

# 训练模型
n_epochs = 5  # 训练轮数
for epoch in range(n_epochs):
    model.train()  # 切换到训练模式
    train_loss = 0.0  # 初始化训练损失
    train_correct = 0  # 初始化训练正确个数
    print(f"Epoch {epoch + 1}/{n_epochs}")  # 打印当前轮数
    print("-" * 40)
    for img, label in train_data_loader:
        img, label = Variable(img), Variable(label)  # 包装成Variable
        outputs = model(img)  # 前向传播
        loss = loss_func(outputs, label)  # 计算损失
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 累加损失
        _, pred = torch.max(outputs.data, 1)  # 获取预测结果
        train_correct += torch.sum(pred == label.data).item()  # 累加正确预测的数量

    # 打印当前轮数的损失和准确率
    print(f"Loss: {train_loss / len(train_data):.4f}, Accuracy: {train_correct / len(train_data):.4f}")
    print()

# 测试模型
model.eval()  # 切换到评估模式
test_correct = 0  # 初始化测试正确个数
with torch.no_grad():  # 不计算梯度
    for img, label in test_data_loader:
        img, label = Variable(img), Variable(label)  # 包装成Variable
        outputs = model(img)  # 前向传播
        _, pred = torch.max(outputs.data, 1)  # 获取预测结果
        test_correct += torch.sum(pred == label.data).item()  # 累加正确预测的数量

# 打印测试准确率
print(f"Test Accuracy: {test_correct / len(test_data):.4f}")

# 保存模型
torch.save(model.state_dict(), "model.pkl")  # 保存模型参数
