import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MLP_model import MLP
from curve import Y

# 设置随机种子，保证结果可重复
torch.manual_seed(128)

# 参数设定
Wmax = 1  # 权重的最大值
lr = 0.2  # 学习率

# 准备 LTP 和 LTD 曲线的电导值
cond = [num for num in Y]
Ptot = int(len(cond) / 2)
cond_max, cond_min = max(cond), min(cond)
cond = [(k - (cond_max + cond_min) / 2) * (Wmax / max(cond)) for k in cond]
LTP, LTD = cond[:Ptot], cond[Ptot:Ptot * 2]
LTP.sort()
LTD.sort()

# 加载 MNIST 数据集进行训练和测试
train_data = torchvision.datasets.MNIST(
    root="data", download=True, train=True, transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root="data", download=True, train=False, transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

# 初始化模型并设置权重
model = MLP()
torch.nn.init.normal_(model.state_dict()['net.1.weight'], mean=0.0, std=0.03)

# 设置损失函数和优化器
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch将学习率衰减为原来的0.1倍

# 设置训练的迭代次数
cnt_epochs = 50
bcc = []
time = []

# 开始训练模型
flag_ltp = 0
flag_ltd = 0

for cnt in range(cnt_epochs):
    model.train()  # 设置模型为训练模式
    for imgs, labels in train_dataloader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(imgs)  # 前向传播
        loss = loss_func(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度

        # 基于计算的梯度更新权重
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    weight = param.data
                    grad = param.grad
                    pulse = (torch.abs(grad) * lr / (Wmax / Ptot / 2)).long()  # 计算脉冲数

                    weight_ltp = weight.clone()  # 复制当前权重
                    weight_ltd = weight.clone()

                    # 长期增强（LTP）情况
                    ltp_mask = grad < 0
                    weight_ltp[ltp_mask] = LTP[torch.clamp(
                        torch.searchsorted(torch.Tensor(LTP), weight_ltp[ltp_mask]) + pulse[ltp_mask], 
                        0, Ptot - 1
                    ).long()]

                    # 长期减弱（LTD）情况
                    ltd_mask = grad > 0
                    weight_ltd[ltd_mask] = LTD[torch.clamp(
                        torch.searchsorted(torch.Tensor(LTD), weight_ltd[ltd_mask]) - pulse[ltd_mask], 
                        0, Ptot - 1
                    ).long()]

                    # 结合更新的权重
                    param.data = torch.where(ltp_mask, weight_ltp, torch.where(ltd_mask, weight_ltd, param.data))

    # 每个 epoch 结束后，使用调度器调整学习率
    scheduler.step()

    # 测试模型在测试集上的表现
    model.eval()  # 设置模型为评估模式
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            outputs = model(imgs)
            predicted = torch.max(outputs, 1)[1]
            correct += (predicted == labels).sum().item()

    # 记录准确率
    acc = correct * 100.0 / len(test_data)
    bcc.append(acc)
    time.append(cnt + 1)
    print(f"Epoch {cnt + 1}, Accuracy: {acc:.2f}%, Current LR: {scheduler.get_last_lr()[0]:.6f}")

# 可视化准确率随 epoch 变化的折线图
plt.figure(figsize=(10, 6))
plt.plot(time, bcc, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Over Epochs')
plt.grid(True)
plt.legend()
plt.show()