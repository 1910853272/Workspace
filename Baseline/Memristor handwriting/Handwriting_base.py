import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MLP_model import MLP  # 导入已经构建好的 MLP 模型。
# from ANN_model import ANN
# from CNN_model import CNN

# 设置随机种子，保证结果可重复
torch.manual_seed(128)

# 第一步：加载 MNIST 数据集进行训练和测试
train_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 第二步：创建 DataLoader 以进行批量处理训练和测试数据集
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
# model = ANN()  # 实例化 MLP 模型
# model = CNN()  # 实例化 MLP 模型

# 第四步：初始化模型、设置损失函数和优化器
model = MLP()
loss_func = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于分类任务
optimizer = optim.SGD(model.parameters(), lr=0.007)  # 使用随机梯度下降优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 第五步：开始训练模型
accuracy_list = []
epoch_list = []
num_epochs = 50

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for batch, (imgs, labels) in enumerate(train_dataloader):
        outputs = model(imgs)  # 前向传播：计算模型输出
        loss = loss_func(outputs, labels)  # 计算损失（交叉熵）
        
        optimizer.zero_grad()  # 清空之前计算的梯度
        loss.backward()  # 反向传播：计算参数的梯度
        optimizer.step()  # 更新参数

    # 每个 epoch 结束后更新学习率
    scheduler.step()

    # 每个 epoch 结束后使用测试数据集评估模型性能
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算以进行评估
        for imgs, labels in test_dataloader:
            outputs = model(imgs)  # 前向传播：获取模型输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确预测的数量

    acc = 100 * correct / total  # 计算准确率百分比
    accuracy_list.append(acc)  # 保存当前 epoch 的准确率
    epoch_list.append(epoch + 1)
    print(f"Epoch {epoch + 1}, Accuracy: {acc:.2f}%")

# 第六步：绘制准确率随训练周期变化的折线图
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Over Epochs')
plt.grid(True)
plt.legend()
plt.show()