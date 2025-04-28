# 引入所需的库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
import os
import utils
import numpy as np
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

# 获取当前时间
t = datetime.fromtimestamp(time.time())

# 设置OECT设备数据的文件名
device_filename = 'oect_data.xlsx'

# 设置保存目录名，并添加当前日期
save_dir_name = 'mnist_0strain_non_resize'
save_dir_name = save_dir_name + '_' + datetime.strftime(t, '%m%d')

# 设置代码的根目录
CODES_DIR = './'
# 数据集路径
DATAROOT = os.path.join(CODES_DIR, 'dataset\\MnistDataset\\processed')

# OECT设备数据的路径
DEVICE_DIR = os.path.join(os.getcwd(), 'data')
device_path = os.path.join(DEVICE_DIR, device_filename)

# 保存结果的路径
SAVE_PATH = 'results'
save_dir_name = os.path.join(SAVE_PATH, save_dir_name)

# 如果目录不存在，则创建相关文件夹
for p in [DATAROOT, SAVE_PATH, save_dir_name]:
    if not os.path.exists(p):
        os.mkdir(p)

# 设置采样频率和其他参数
sampling = 0
num_pulse = 4  # 这里的注释说明原来为5
num_pixels = 28 * 28
new_img_width = 196
batchsize = 1
device_tested_number = 2  # 设定测试的设备数量

digital = False  # 是否进行数字化处理

# 设置训练和测试的批量大小
batchsize = 128
te_batchsize = 128
oect_tested_number = 4

# 设置训练和测试数据的路径
TRAIN_PATH = os.path.join(DATAROOT, 'training.pt')
TEST_PATH = os.path.join(DATAROOT, 'test.pt')

# 加载训练和测试数据集
tr_dataset = utils.SimpleDataset(TRAIN_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)
te_dataset = utils.SimpleDataset(TEST_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)

# 创建数据加载器
train_loader = DataLoader(tr_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(te_dataset, batch_size=batchsize)

# 设置训练参数
num_epoch = 30  # 训练轮数
learning_rate = 1e-2  # 学习率

# 获取训练集和测试集的数据量
num_data = len(tr_dataset)
num_te_data = len(te_dataset)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义模型
model = torch.nn.Sequential(
    nn.Linear(new_img_width, 10)  # 全连接层，输入特征为new_img_width，输出为10个类别
)

# 使用Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # 学习率衰减

# 处理OECT设备数据
device_output = utils.oect_data_proc(path=device_path, device_test_cnt=device_tested_number, device_read_times=None)

# 如果进行数字化处理，则对数据进行相应的转换
if digital:
    d_outputs = np.arange(2 ** num_pulse) / (2 ** num_pulse - 1)
    device_output = device_output[1]
    device_output[:] = np.arange(2 ** num_pulse)
else:
    # 0-1归一化处理OECT设备数据
    device_output = (device_output - device_output.min().min()) / (device_output.max().max() - device_output.min().min())

# 开始训练
start_time = time.time()
acc_list = []
loss_list = []
print("Start training...")

# 训练循环
for epoch in range(num_epoch):
    acc = []  # 存储每个batch的准确率
    loss = 0  # 存储每个epoch的总损失
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 清除之前的梯度

        this_batch_size = len(data)

        # 提取OECT特征
        oect_output = utils.batch_rc_feat_extract(data, device_output, device_tested_number, num_pulse, this_batch_size)

        # 通过模型进行前向传播
        logic = model(oect_output)

        # 计算损失
        batch_loss = criterion(logic, target)
        loss += batch_loss

        # 计算准确率
        batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
        acc.append(batch_acc)

        # 反向传播并更新权重
        batch_loss.backward()
        optimizer.step()

    # 更新学习率
    scheduler.step()

    # 计算本轮训练的平均准确率
    acc_epoch = (sum(acc) * batchsize / num_data).numpy()
    acc_list.append(acc_epoch)
    loss_list.append(loss)

    # 计算每轮的训练时间
    epoch_end_time = time.time()
    if epoch == 0:
        epoch_time = epoch_end_time - start_time
    else:
        epoch_time = epoch_end_time - epoch_start_time
    epoch_start_time = epoch_end_time

    print("epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time))

# 在测试集上进行评估
te_accs = []
te_outputs = []
targets = []
with torch.no_grad():  # 在测试时不计算梯度
    for i, (data, target) in enumerate(test_dataloader):

        this_batch_size = len(data)
        # 提取测试数据的OECT特征
        oect_output = utils.batch_rc_feat_extract(data, device_output, device_tested_number, num_pulse, this_batch_size)

        # 获取模型输出
        output = model(oect_output)

        # 计算准确率
        te_outputs.append(output)
        acc = torch.sum(output.argmax(dim=-1) == target) / te_batchsize
        te_accs.append(acc)
        targets.append(target)

    # 计算测试准确率
    te_acc = (sum(te_accs) * te_batchsize / num_te_data).numpy()
    print("test acc: %.6f" % te_acc)

    # 拼接所有输出和目标
    te_outputs = torch.cat(te_outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # 计算混淆矩阵
    conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=1))

    # 转换为DataFrame格式并进行归一化
    conf_mat_dataframe = pd.DataFrame(conf_mat, index=list(range(10)), columns=list(range(10)))
    conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

    # 可视化混淆矩阵
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mat_dataframe, annot=True, fmt='d')
    plt.savefig(os.path.join(save_dir_name, 'conf_mat'))
    plt.close()

    # 可视化归一化后的混淆矩阵
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mat_normalized, annot=True)
    plt.savefig(os.path.join(save_dir_name, 'conf_mat_normlized'))
    plt.close()

    # 保存模型
    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))
