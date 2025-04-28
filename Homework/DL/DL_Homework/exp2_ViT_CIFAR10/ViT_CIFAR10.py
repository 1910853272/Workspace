# -*- coding: utf-8 -*-
import time  # 用于计算训练耗时

import torch
from torch import nn
import torchvision  # torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from einops import rearrange, repeat  # 高效张量维度变换库
from einops.layers.torch import Rearrange  # 用于构建可插拔的维度重排层

# ==================== 数据预处理设置 ====================
<<<<<<< HEAD
=======
# 训练集数据增强：随机裁剪、随机水平翻转、归一化
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 在 32x32 图片上随机裁剪并在边缘填充 4 像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转为 [0,1] 之间的张量
<<<<<<< HEAD
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=trans_valid)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
=======
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化到预训练模型常用均值
                         std=[0.229, 0.224, 0.225])  # 对应的方差
])

# 验证/测试集预处理：只做归一化
trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 准备 CIFAR-10 测试集及加载器
testset = torchvision.datasets.CIFAR10(
    root="data", train=False, download=False, transform=trans_valid)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
# CIFAR-10 的十个类别名称
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==================== 多头自注意力模块 ====================
class Attention(nn.Module):
<<<<<<< HEAD
    def __init__(self, dim=128, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
=======
    """
    多头自注意力层，包含可选投影输出和残差连接。
    """
    def __init__(self, dim=128, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads  # 总的头维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要线性投影回原维度

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，避免内积过大
        self.norm = nn.LayerNorm(dim)  # 对输入做 LayerNorm
        self.attend = nn.Softmax(dim=-1)  # 注意力权重的 Softmax
        self.dropout = nn.Dropout(dropout)  # 注意力权重后的 Dropout
        # linear 用于生成 q, k, v 三个矩阵
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # 若需要，将多头拼接后的输出投影回 dim 维
        self.to_out = (nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity())

    def forward(self, x):
        # x: [batch, seq_len, dim]
        x = self.norm(x)  # 先做 LayerNorm
        # to_qkv 之后切分成 q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 将每个头的维度分离
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数：q·k^T
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, heads, n, n]
        attn = self.attend(dots)  # 归一化注意力权重
        attn = self.dropout(attn)  # Dropout

        # 加权求和值
        out = torch.matmul(attn, v)  # [b, heads, n, dim_head]
        # 将多头拼回去：合并头维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 最后一层线性投影（若有）
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        return self.to_out(out)

# ==================== 前馈网络 ====================
class FeedForward(nn.Module):
<<<<<<< HEAD
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
=======
    """
    标准 MLP 前馈网络，包含两层全连接和激活。
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # LayerNorm
            nn.Linear(dim, hidden_dim),  # 第一层全连接
            nn.GELU(),  # GELU 激活
            nn.Dropout(dropout),  # Dropout
            nn.Linear(hidden_dim, dim),  # 第二层全连接投回原维度
            nn.Dropout(dropout)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        )

    def forward(self, x):
        return self.net(x)

# ==================== Transformer 编码器 ====================
class Encoder(nn.Module):
<<<<<<< HEAD
    def __init__(self, dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
=======
    """
    多层堆叠的 Transformer 编码器，每层包含自注意力和前馈网络。
    """
    def __init__(self, dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 最终归一化
        self.layers = nn.ModuleList([])
        # 逐层堆叠
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
<<<<<<< HEAD
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# ==================== Vision Transformer (ViT) ====================
class ViT(nn.Module):
=======
        # 每一层：注意力 + 残差，前馈 + 残差
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x
        return self.norm(x)  # 最后做一次 LayerNorm

# ==================== Vision Transformer (ViT) ====================
class ViT(nn.Module):
    """
    Vision Transformer 主体，用于图像分类。
    """
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    def __init__(self, num_classes=10, dim=512, depth=6, heads=8,
                 mlp_dim=512, pool='cls', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
<<<<<<< HEAD
        image_height, image_width = 32, 32
        patch_height, patch_width = 4, 4
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

# ==================== 训练与测试函数 ====================
def train(epoch):
    print(f"训练 {epoch} 轮开始...")
    model = ViT()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    loss_all = []
    acc_all = []
=======
        # 图像尺寸及分片参数
        image_height, image_width = 32, 32
        patch_height, patch_width = 4, 4
        # 计算 patch 数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width  # 每个 patch 的像素展开维度

        # patch embedding: 将图像切片并线性映射到 dim 维特征
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # 对 patch 数据归一化
            nn.Linear(patch_dim, dim),  # 线性映射到 dim
            nn.LayerNorm(dim),  # 再次归一化
        )

        # 可训练的位置嵌入，以及分类 token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer 编码器
        self.transformer = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool  # 池化方式：'cls' 或 'mean'
        self.to_latent = nn.Identity()  # 占位
        # 最终分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # 输入 img: [batch, channels, H, W]
        x = self.to_patch_embedding(img)  # patch embedding -> [b, num_patches, dim]
        b, n, _ = x.shape
        # 扩展 cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        # 拼接 cls token
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, dim]
        # 加入位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer 编码
        x = self.transformer(x)  # [b, n+1, dim]
        # 池化取输出：分类 token 或 均值
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)  # 最终分类输出

# ==================== 训练与测试函数 ====================
def train(epoch):
    """
    训练函数：完成多轮训练并保存模型与日志，同时可视化损失和准确率曲线。
    """
    print(f"训练 {epoch} 轮开始...")
    # 实例化模型和设备
    model = ViT()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Adam 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    loss_all = []  # 记录每轮训练损失
    acc_all = []  # 记录每轮测试准确率
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    start_time = time.time()

    for e in range(epoch):
        net.train()
<<<<<<< HEAD
        trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=trans_train)
=======
        # 加载训练集
        trainset = torchvision.datasets.CIFAR10(
            root="data", train=True, download=False, transform=trans_train)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

<<<<<<< HEAD
        avg_loss = running_loss / len(trainloader)
        loss_all.append(avg_loss)
=======
        # 计算平均损失
        avg_loss = running_loss / len(trainloader)
        loss_all.append(avg_loss)
        # 在测试集上评估准确率
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        acc = test(net, device, criterion)
        acc_all.append(acc)
        print(f"轮 {e+1}/{epoch} - 损失: {avg_loss:.4f} - 准确率: {acc:.4f} - 用时: {time.time()-start_time:.1f}s")
        start_time = time.time()

<<<<<<< HEAD
=======
    # 保存训练日志和模型参数
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    with open('loss.txt', 'w', encoding="utf-8") as f:
        f.write(str(loss_all))
    with open('acc.txt', 'w', encoding="utf-8") as f:
        f.write(str(acc_all))
    torch.save(model.state_dict(), './model1.pt')

<<<<<<< HEAD
    # 可视化并保存训练损失曲线
=======
    # 可视化训练损失曲线
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    epochs = list(range(1, epoch + 1))
    plt.figure()
    plt.plot(epochs, loss_all, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
<<<<<<< HEAD
    plt.savefig('train_loss.png')  # 保存为图像文件
    plt.show()

    # 可视化并保存测试准确率曲线
=======
    plt.savefig('Train_Loss.png')  # 保存为图像文件
    plt.show()

    # 可视化测试准确率曲线
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    plt.figure()
    plt.plot(epochs, acc_all, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
<<<<<<< HEAD
    plt.savefig('train_accuracy.png')  # 保存为图像文件
=======
    plt.savefig('Train_Accuracy.png')  # 保存为图像文件
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    plt.show()


def test(net, device, criterion):
<<<<<<< HEAD
=======
    """
    在测试集上评估模型准确率。
    """
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

<<<<<<< HEAD
if __name__ == '__main__':
    train(200)
=======
# ========== 主流程 ==========
if __name__ == '__main__':
    # 执行训练
    train(300)

    # 训练完成后加载模型并评估测试准确率
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViT()
    model.load_state_dict(torch.load('./model1.pt', map_location=device))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    final_acc = test(model, device, criterion)
    print(f"最终测试集准确率: {final_acc:.4f}")
