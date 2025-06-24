import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 模拟忆阻器的电导行为
def output_row_memristor(initial_state, input_signal):
    """
    模拟memristor的行为，生成电导值的序列。
    """
    a = [initial_state]
    for i in range(len(input_signal)):
        if input_signal[i] > 0:
            a.append(np.clip(a[i], 0.1, 1) * np.exp(1))
        else:
            a.append(np.clip(a[i], 1, 10) * (3 - np.exp(1)))
    return np.array(a).flatten()

# 创建输入数据
letters = ['l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy',
           'l5_p.npy', 'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy']
d = {}  # 用来存储字母与电导值的映射
for lett in letters:
    # 加载文件并生成电导值
    data = np.load(lett)
    initial_state = np.random.random(1)
    output = [output_row_memristor(initial_state, row) for row in data]
    d[lett] = np.concatenate(output)

# 输入矩阵（电导序列）
X = np.zeros((10, 30))  # 假设每个字母有5行数据，每行6个特征，总共10个字母
for i, letter in enumerate(letters):
    X[i, :] = d[letter]

# 标签
y = np.arange(10)  # 每个字母对应一个标签

# 转换为Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 定义简单的神经网络（只有一层）
class MemristorNN(nn.Module):
    def __init__(self):
        super(MemristorNN, self).__init__()
        self.fc1 = nn.Linear(30, 10)  # 30个特征，10个类别

    def forward(self, x):
        x = self.fc1(x)  # 只有一个线性层
        return x

# 初始化模型、损失函数和优化器
model = MemristorNN()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 简单的梯度下降优化

# 训练模型
epochs = 50
losses = []
accuracies = []

for epoch in range(epochs):
    model.train()

    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率
    _, predicted_labels = torch.max(outputs, 1)  # 获取预测的标签
    correct = (predicted_labels == y_tensor).sum().item()  # 计算正确预测的数量
    accuracy = correct / len(y_tensor)  # 计算准确率
    accuracies.append(accuracy)

    losses.append(loss.item())  # 记录损失

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%")

# 预测
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    _, predicted_labels = torch.max(predictions, 1)

# 打印混淆矩阵
confusion_matrix = np.zeros((10, 10))
for true, pred in zip(y_tensor, predicted_labels):
    confusion_matrix[true.item(), pred.item()] += 1

# 可视化损失曲线
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# 可视化准确率曲线
plt.figure()
plt.plot(accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 可视化混淆矩阵
plt.figure()
plt.imshow(confusion_matrix, cmap="viridis")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
