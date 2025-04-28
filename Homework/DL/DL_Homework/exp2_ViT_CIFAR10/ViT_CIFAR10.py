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
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 在 32x32 图片上随机裁剪并在边缘填充 4 像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转为 [0,1] 之间的张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=trans_valid)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==================== 多头自注意力模块 ====================
class Attention(nn.Module):
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
        return self.to_out(out)

# ==================== 前馈网络 ====================
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# ==================== Transformer 编码器 ====================
class Encoder(nn.Module):
    def __init__(self, dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# ==================== Vision Transformer (ViT) ====================
class ViT(nn.Module):
    def __init__(self, num_classes=10, dim=512, depth=6, heads=8,
                 mlp_dim=512, pool='cls', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
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
    start_time = time.time()

    for e in range(epoch):
        net.train()
        trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=trans_train)
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

        avg_loss = running_loss / len(trainloader)
        loss_all.append(avg_loss)
        acc = test(net, device, criterion)
        acc_all.append(acc)
        print(f"轮 {e+1}/{epoch} - 损失: {avg_loss:.4f} - 准确率: {acc:.4f} - 用时: {time.time()-start_time:.1f}s")
        start_time = time.time()

    with open('loss.txt', 'w', encoding="utf-8") as f:
        f.write(str(loss_all))
    with open('acc.txt', 'w', encoding="utf-8") as f:
        f.write(str(acc_all))
    torch.save(model.state_dict(), './model1.pt')

    # 可视化并保存训练损失曲线
    epochs = list(range(1, epoch + 1))
    plt.figure()
    plt.plot(epochs, loss_all, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.savefig('train_loss.png')  # 保存为图像文件
    plt.show()

    # 可视化并保存测试准确率曲线
    plt.figure()
    plt.plot(epochs, acc_all, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    plt.savefig('train_accuracy.png')  # 保存为图像文件
    plt.show()


def test(net, device, criterion):
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

if __name__ == '__main__':
    train(200)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViT()
    model.load_state_dict(torch.load('./model1.pt', map_location=device))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    final_acc = test(model, device, criterion)
    print(f"最终测试集准确率: {final_acc:.4f}")
