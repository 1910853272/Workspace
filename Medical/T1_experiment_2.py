import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, \
    precision_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.manifold import TSNE

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]

# 1. 读取数据
data = pd.read_csv('reshaped_T1_features2.csv')

# 2. 数据预处理 - 保留空间结构，只使用前90个regions
feature_cols = [col for col in data.columns if col not in ['subject_id', 'group', 'modality', 'region_id',
                                                           'group_AD', 'group_MCI', 'group_NC']]

# 获取region数量和特征维度
n_regions = min(data['region_id'].nunique(), 90)  # 限制最多使用90个regions
n_features = len(feature_cols)

print(f"使用 {n_regions} 个regions和 {n_features} 个特征")

# 按 subject_id 分组，构建空间特征张量
grouped = data.groupby('subject_id')
sample_features = []
sample_labels = []

for subject, group in grouped:
    # 确保每个subject的region顺序一致，并只取前90个regions
    group = group.sort_values('region_id').head(n_regions)

    # 如果region不足90个，则填充零
    region_features = np.zeros((n_regions, n_features))
    actual_regions = min(n_regions, len(group))
    region_features[:actual_regions] = group[feature_cols].values

    sample_features.append(region_features)
    label = group['group'].unique()[0]
    sample_labels.append(label)

# 转换为 numpy 数组
X = np.array(sample_features)  # 形状: (n_samples, n_regions, n_features)
y = np.array(sample_labels)

# 3. 定义三个二分类任务
binary_tasks = {
    'AD_vs_NotAD': {'positive_class': 'AD', 'negative_class': ['MCI', 'NC']},
    'MCI_vs_NotMCI': {'positive_class': 'MCI', 'negative_class': ['AD', 'NC']},
    'NC_vs_NotNC': {'positive_class': 'NC', 'negative_class': ['AD', 'MCI']}
}


# 4. 创建PyTorch数据集
class BrainDataset(Dataset):
    def __init__(self, features, labels, scaler=None):
        self.features = features
        self.labels = labels

        # 特征标准化
        if scaler is None:
            self.scaler = StandardScaler()
            # 展平进行标准化
            n_samples, n_regions, n_features = self.features.shape
            flat_features = self.features.reshape(n_samples * n_regions, n_features)
            scaled_flat = self.scaler.fit_transform(flat_features)
            self.features = scaled_flat.reshape(n_samples, n_regions, n_features)
        else:
            self.scaler = scaler
            n_samples, n_regions, n_features = self.features.shape
            flat_features = self.features.reshape(n_samples * n_regions, n_features)
            scaled_flat = self.scaler.transform(flat_features)
            self.features = scaled_flat.reshape(n_samples, n_regions, n_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 添加空间维度，将 (n_regions, n_features) 转换为 (1, n_regions, n_features)
        # 对于90个regions，我们可以使用 5x5x4 的三维结构 (5*5*4=100)
        spatial_dim_x = 5
        spatial_dim_y = 5
        spatial_dim_z = 4

        # 填充到合适的维度
        feature_tensor = np.zeros((spatial_dim_x, spatial_dim_y, spatial_dim_z, n_features))
        for i in range(min(n_regions, spatial_dim_x * spatial_dim_y * spatial_dim_z)):
            x = i % spatial_dim_x
            y = (i // spatial_dim_x) % spatial_dim_y
            z = i // (spatial_dim_x * spatial_dim_y)
            feature_tensor[x, y, z] = self.features[idx, i]

        # 转换为PyTorch张量并调整维度为 (C, D, H, W)
        feature_tensor = torch.tensor(feature_tensor, dtype=torch.float32).permute(3, 2, 1, 0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature_tensor, label

class Brain3DCNN(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super(Brain3DCNN, self).__init__()

        # 3D卷积层
        self.conv1 = nn.Conv3d(n_features, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 自动计算全连接层输入大小
        dummy_input = torch.rand(1, n_features, 4, 5, 5)  # 对应 (C, D, H, W)
        fc_input_size = self._get_flattened_size(dummy_input)

        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

        # 特征重要性权重
        self.feature_importance = nn.Parameter(torch.ones(n_features) / n_features)

        # 区域重要性映射
        self.region_importance = nn.Parameter(torch.ones(n_regions) / n_regions)

    def _get_flattened_size(self, x):
        """计算卷积层输出的展平尺寸"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        # 应用特征重要性权重
        x = x * self.feature_importance.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 卷积层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# 6. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=200):
    best_val_auc = 0.0
    best_model = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1]

                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_probs)
        avg_train_loss = running_loss / len(train_loader.dataset)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model


# 7. 执行三个二分类实验（带5折交叉验证）
results = {}

for task_name, task_config in binary_tasks.items():
    print(f"\n===== 正在进行二分类实验: {task_name} =====")

    positive_class = task_config['positive_class']
    negative_classes = task_config['negative_class']

    # 创建二分类标签
    class_mapping = {positive_class: 1}
    for neg_class in negative_classes:
        class_mapping[neg_class] = 0

    y_binary = np.array([class_mapping[label] for label in y])

    # 初始化指标存储
    metrics = {
        'auc': [], 'acc': [], 'tpr': [], 'fpr': [],
        'precision': [], 'f1': [], 'kappa': []
    }

    # 特征重要性存储
    feature_importance_list = []
    region_importance_list = []

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_binary)):
        print(f"\n===== 第 {fold_idx + 1}/5 折 =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # 创建数据集和数据加载器
        train_dataset = BrainDataset(X_train, y_train)
        test_dataset = BrainDataset(X_test, y_test, scaler=train_dataset.scaler)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 初始化模型
        model = Brain3DCNN(n_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30)

        # 评估模型
        model.eval()
        test_labels = []
        test_probs = []
        test_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

        # 计算各项指标
        auc = roc_auc_score(test_labels, test_probs)
        acc = accuracy_score(test_labels, test_preds)
        tpr = recall_score(test_labels, test_preds)  # TPR = Recall = TP / (TP + FN)
        fpr = fp / (fp + tn)  # FPR = FP / (FP + TN)
        precision = precision_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        kappa = cohen_kappa_score(test_labels, test_preds)

        # 存储指标
        metrics['auc'].append(auc)
        metrics['acc'].append(acc)
        metrics['tpr'].append(tpr)
        metrics['fpr'].append(fpr)
        metrics['precision'].append(precision)
        metrics['f1'].append(f1)
        metrics['kappa'].append(kappa)

        # 提取特征重要性
        feature_importance = model.feature_importance.detach().cpu().numpy()
        feature_importance_list.append(feature_importance)

        # 提取区域重要性
        region_importance = model.region_importance.detach().cpu().numpy()
        region_importance_list.append(region_importance)

        print(f"第 {fold_idx + 1}/5 折: AUC={auc:.4f}, ACC={acc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}, "
              f"Precision={precision:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

    # 计算平均指标
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    std_metrics = {metric: np.std(values) for metric, values in metrics.items()}

    # 保存结果
    results[task_name] = {
        'metrics': metrics,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'feature_importance': np.mean(feature_importance_list, axis=0),
        'region_importance': np.mean(region_importance_list, axis=0)
    }

    # 打印最终指标
    print(f"\n{task_name} 5折交叉验证平均指标:")
    for metric in ['auc', 'acc', 'tpr', 'fpr', 'precision', 'f1', 'kappa']:
        print(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# 8. 绘制三个任务的指标对比图
metrics_to_plot = ['auc', 'acc', 'tpr', 'fpr', 'precision', 'f1', 'kappa']
tasks = list(results.keys())

plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 4, i)
    task_values = [results[task]['avg_metrics'][metric] for task in tasks]
    task_stds = [results[task]['std_metrics'][metric] for task in tasks]

    plt.bar(tasks, task_values, yerr=task_stds, capsize=5)
    plt.title(metric.upper())
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('binary_classification_metrics_comparison.png')
plt.show()

# 9. 可视化特征重要性
plt.figure(figsize=(15, 10))
for i, task in enumerate(tasks, 1):
    plt.subplot(1, 3, i)
    importance = results[task]['feature_importance']
    indices = np.argsort(importance)[::-1][:10]  # 取前10个重要特征

    plt.barh(range(10), importance[indices], align='center')
    plt.yticks(range(10), [feature_cols[i] for i in indices])
    plt.xlabel('重要性')
    plt.title(f'{task} - 特征重要性')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.show()

# 10. 可视化区域重要性
plt.figure(figsize=(15, 10))
for i, task in enumerate(tasks, 1):
    plt.subplot(1, 3, i)
    importance = results[task]['region_importance']
    indices = np.argsort(importance)[::-1][:10]  # 取前10个重要区域

    plt.barh(range(10), importance[indices], align='center')
    plt.yticks(range(10), [f'Region {idx}' for idx in indices])
    plt.xlabel('重要性')
    plt.title(f'{task} - 区域重要性')

plt.tight_layout()
plt.savefig('region_importance_comparison.png')
plt.show()