# 导入必要的库，包括PyTorch、torchvision、数据加载工具和自定义的utils模块
import torch
from torch.functional import split
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import utility.utils as utils  # 自定义的工具模块
import os

# 解析从命令行传递的选项
options = utils.parse_args()

# 设置训练批次大小
batchsize = options.batch
te_batchsize = options.batch  # 测试批次大小与训练批次大小相同
CODES_DIR = os.path.dirname(os.getcwd())  # 获取当前工作目录的父目录
# 数据集的根目录
DATAROOT = os.path.join(CODES_DIR, 'MNIST_CLS/data/MNIST/processed')  # MNIST处理后数据的路径
DATAROOT = 'dataset/EMNIST/bymerge/processed'  # EMNIST的数据集路径
# OECT数据路径
DEVICE_DIR = os.path.join(os.getcwd(), 'data')

# 训练集和测试集的路径
TRAIN_PATH = os.path.join(DATAROOT, 'training_bymerge.pt')
TEST_PATH = os.path.join(DATAROOT, 'test_bymerge.pt')

# 加载训练数据集并进行预处理
tr_dataset = utils.SimpleDataset(TRAIN_PATH,
                                 num_pulse=options.num_pulse,
                                 crop=options.crop,
                                 sampling=options.sampling,
                                 ori_img=True)
# 加载测试数据集并进行预处理
te_dataset = utils.SimpleDataset(TEST_PATH,
                                 num_pulse=options.num_pulse,
                                 crop=options.crop,
                                 sampling=options.sampling,
                                 ori_img=True)

# 定义一个简单的神经网络模型
# 输入层784（28x28像素的扁平化），输出层47个分类（对应EMNIST bymerge数据集的类数）
model = torch.nn.Sequential(
    nn.Linear(784, 47)  # 全连接层，输入为784，输出为47
)

# 使用DataLoader加载训练数据集，设置批次大小并打乱数据顺序
train_loader = DataLoader(tr_dataset,
                          batch_size=batchsize,
                          shuffle=True)
# 使用DataLoader加载测试数据集
test_dataloader = DataLoader(te_dataset, batch_size=batchsize)

# 设置训练的超参数
num_epoch = 50  # 训练的轮次
learning_rate = 1e-3  # 学习率

# 获取训练集和测试集的样本数量
num_data = len(tr_dataset)
num_te_data = len(te_dataset)

# 使用交叉熵损失函数作为分类任务的损失函数
criterion = nn.CrossEntropyLoss()

# 使用Adam优化器更新模型参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 使用学习率调度器，每15轮将学习率乘以0.1，逐渐减小学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 训练部分
acc_list = []  # 用于保存每轮训练的准确率
loss_list = []  # 用于保存每轮训练的损失
for epoch in range(num_epoch):  # 遍历每个epoch

    acc = []  # 保存每个批次的准确率
    loss = 0  # 保存每个epoch的总损失
    for i, (data, _, target) in enumerate(train_loader):  # 遍历训练数据的每个批次
        optimizer.zero_grad()  # 每个批次前清除梯度

        this_batch_size = len(data)  # 当前批次的数据大小

        data = data.to(torch.float)  # 将数据类型转换为浮点型
        # 使用模型预测
        logic = model(data)
        logic = torch.squeeze(logic)  # 压缩输出以去除多余的维度

        # 计算当前批次的损失
        batch_loss = criterion(logic, target)
        loss += batch_loss  # 累积损失
        # 计算当前批次的准确率
        batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
        acc.append(batch_acc)  # 保存准确率
        batch_loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

    scheduler.step()  # 调度学习率
    # 计算并保存整个epoch的准确率和损失
    acc_epoch = (sum(acc) * batchsize / num_data).numpy()  # 将准确率转换为numpy格式
    acc_list.append(acc_epoch)  # 保存每轮的准确率
    loss_list.append(loss)  # 保存每轮的损失

    # 打印当前轮次的损失和准确率
    print("epoch: %d, loss: %.2f, acc: %.6f, " % (epoch, loss, acc_epoch))

# 测试部分
te_accs = []  # 保存每个批次的测试准确率
te_outputs = []  # 保存模型的输出
targets = []  # 保存真实标签
with torch.no_grad():  # 测试时不计算梯度
    for i, (data, target) in enumerate(test_dataloader):  # 遍历测试数据集的每个批次

        this_batch_size = len(data)  # 获取当前批次的大小
        output = model(data.to(torch.float))  # 使用模型进行预测
        output = torch.squeeze(output)  # 压缩输出以去除多余的维度
        te_outputs.append(output)  # 保存预测输出
        # 计算当前批次的准确率
        acc = torch.sum(output.argmax(dim=-1) == target) / te_batchsize
        te_accs.append(acc)  # 保存每个批次的准确率
        targets.append(target)  # 保存目标标签

    # 计算整个测试集的准确率
    te_acc = (sum(te_accs) * te_batchsize / num_te_data).numpy()
    print("test acc: %.6f" % te_acc)  # 打印测试集的准确率
