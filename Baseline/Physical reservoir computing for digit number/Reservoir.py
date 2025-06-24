import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
data_path = 'dataset.xlsx'  # 确保路径正确
df = pd.read_excel(data_path)

# 2. 数据预处理
# 2.1 分离特征和标签
# 假设第一列是 'label'，接下来的80列是特征
features = df.iloc[:, 1:].values  # 获取特征数据，形状为 (5000, 80)
labels = df.iloc[:, 0].values     # 获取标签数据，形状为 (5000,)

# 2.2 编码标签
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # 将 'digit_0' 到 'digit_9' 编码为 0 到 9

# 2.3 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # 对特征进行标准化

# 3. 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 训练特征
y_train_tensor = torch.tensor(y_train, dtype=torch.long)     # 训练标签
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)    # 测试特征
y_test_tensor = torch.tensor(y_test, dtype=torch.long)       # 测试标签

# 4. 创建自定义数据集类
class DigitDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建训练集和测试集的数据集对象
train_dataset = DigitDataset(X_train_tensor, y_train_tensor)
test_dataset = DigitDataset(X_test_tensor, y_test_tensor)

# 创建数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # 测试数据加载器

# 5. 定义感知机模型
class PerceptronModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PerceptronModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 线性层

    def forward(self, x):
        out = self.linear(x)  # 前向传播
        return out

input_dim = X_train.shape[1]  # 输入特征维度为80
output_dim = len(np.unique(labels_encoded))  # 输出类别数为10

model = PerceptronModel(input_dim, output_dim)  # 实例化模型

# 6. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类
optimizer = optim.SGD(model.parameters(), lr=0.001)  # 随机梯度下降优化器，学习率为0.001

# 7. 训练模型
num_epochs = 100  # 训练轮数
train_accuracy_history = []  # 用于记录每轮的训练准确率

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    correct = 0
    total = 0
    epoch_loss = 0

    for batch_features, batch_labels in train_loader:
        # 前向传播
        outputs = model(batch_features)  # 获取模型输出
        loss = criterion(outputs, batch_labels)  # 计算损失

        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        epoch_loss += loss.item()  # 累加损失

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += batch_labels.size(0)               # 累加样本数量
        correct += (predicted == batch_labels).sum().item()  # 累加正确预测的数量

    # 计算本轮的准确率
    accuracy = correct / total
    train_accuracy_history.append(accuracy)  # 记录准确率

    # 每10轮打印一次损失和准确率，准确率以百分数显示
    if (epoch+1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')

# 8. 绘制准确率与轮数的关系
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [acc * 100 for acc in train_accuracy_history], label='Training Accuracy')
plt.xlabel('Epoch')  # 横轴为轮数
plt.ylabel('Accuracy (%)')  # 纵轴为准确率，单位为百分比
plt.title('Perceptron Training Accuracy over Epochs')  # 图表标题
plt.legend()
plt.grid(True)
plt.show()

# 9. 在测试集上评估模型
model.eval()  # 设置模型为评估模式
all_preds = []
all_labels = []

with torch.no_grad():  # 不计算梯度
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)  # 获取模型输出
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        all_preds.extend(predicted.numpy())  # 记录预测结果
        all_labels.extend(batch_labels.numpy())  # 记录真实标签

# 计算测试集准确率，并以百分比显示
test_accuracy = accuracy_score(all_labels, all_preds)
print(f'\nPerceptron Model Test Accuracy: {test_accuracy*100:.2f}%')

# 绘制归一化后的混淆矩阵，突出显示每个类别的准确率，数值以百分比显示
plt.figure(figsize=(10, 8))
conf_matrix_normalized = confusion_matrix(all_labels, all_preds, normalize='true') * 100  # 归一化并转换为百分比
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.xlabel('Predicted Label (%)')  # X轴标签
plt.ylabel('True Label (%)')       # Y轴标签
plt.title('Perceptron Model Confusion Matrix (Normalized %)')  # 图表标题
plt.show()
