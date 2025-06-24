import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

# 固定随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((168, 168)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建保存图像的文件夹
output_dir = 'plot'
os.makedirs(output_dir, exist_ok=True)

# 定义模型
class MLP(nn.Module):
    def __init__(self, input_dim=168*168, hidden_dim1=64, hidden_dim2=16, num_classes=6):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 20
test_acc_history = []

# 训练与评估
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        preds, true_labels = [], []
        for images, labels in test_loader:
            images = images.to(device)
            output = model(images)
            predicted = torch.argmax(output, dim=1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())

        acc = accuracy_score(true_labels, preds)
        test_acc_history.append(acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {acc*100:.2f}%")

# 准确率曲线图
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), [acc*100 for acc in test_acc_history], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs Epoch')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
plt.close()

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, preds)
classes = train_dataset.classes

def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

plot_confusion_matrix(cm, classes=classes)

print(f"所有图像已保存到: {output_dir}")
