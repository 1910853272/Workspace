import os
import sys
import torch
from torch import nn
import numpy as np
from utility.dvs_dataset import DvsDataset
import matplotlib.pyplot as plt

# 定义基础目录，用于加载自定义模块和数据集
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 设置参数
start_f = 6  # 从第 6 帧开始选择帧
num_frame_split = 32  # 总共将点云数据分为 32 帧
img_width = 28  # 图像的宽度（28x28 像素）

# 加载数据集，分别加载训练集和测试集
train_dataset = DvsDataset(DATADIR='dataset/DVS_C10_TS1_1024', train=True, use_raw=False)
test_dataset = DvsDataset(DATADIR='dataset/DVS_C10_TS1_1024', train=False, use_raw=False)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

# 定义保存目录，如果目录不存在，则创建
SAVE_DIR = 'dataset/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
# 为 11 个类别（从 0 到 10）创建子目录
for i in range(11):
    subdir = os.path.join(SAVE_DIR, str(i))
    if not os.path.exists(subdir):
        os.mkdir(subdir)

# 遍历训练和测试数据集的加载器
for tp, Loader in zip(['tr', 'te'], [trainDataLoader, testDataLoader]):
    processed_dataset, labels = [], []
    
    # 遍历每个样本，data 是事件数据，label 是对应的类别标签
    for i, (data, label) in enumerate(Loader):
        
        # 仅处理属于类别 0 到 4 的样本
        if label.item() in [0, 1, 2, 3, 4]:
            data = data[0]  # 获取样本的事件数据
            print(f'label: {label}, data shape: {data.shape}')  # 输出标签和数据形状

            # 按时间排序点云数据，第 0 轴是时间轴
            cloud = data[data[:, 0].argsort()]

            # 提取坐标并归一化到 [0, 28] 区间
            cloud_coord = cloud[:, 1:]
            cloud_coord = (cloud_coord - cloud_coord.min()) / (cloud_coord.max() - cloud_coord.min()) * img_width
            cloud_coord = torch.where(cloud_coord == img_width, torch.tensor(img_width - 1).to(cloud_coord.dtype), cloud_coord) 
            cloud[:, 1:] = cloud_coord.to(torch.int)

            # 将点云数据分成多个帧
            cloud_new_idx = list(torch.split(cloud, int(1024 / num_frame_split)))
            images = torch.zeros((num_frame_split, img_width, img_width))
            
            # 为每个点分配帧索引
            for i in range(num_frame_split):
                cloud_new_idx[i] = torch.cat((torch.ones_like(cloud_new_idx[i].unsqueeze(-1))[:, 0] * i, cloud_new_idx[i][:, 1:]), dim=1)

            # 合并所有帧的点数据
            cloud_new_idx = torch.cat(cloud_new_idx, dim=0)
            cloud_new_idx = cloud_new_idx.to(torch.long)

            # 将每个事件点放入对应帧的图像中
            for point in cloud_new_idx:
                images[point[0], point[1], point[2]] = 1

            # 从第 6 帧开始，每隔 5 帧选择一个，总共选择 5 帧
            images_5frame = images[start_f: :5][:5]

            # 保存处理好的 5 帧数据和对应的标签
            processed_dataset.append(images_5frame)
            labels.append(label)

    # 将数据转换为张量并保存到文件
    processed_dataset = torch.stack(processed_dataset)
    labels = torch.tensor(labels)
    torch.save((processed_dataset, labels), os.path.join(SAVE_DIR, f'dvs_proced_{tp}_5cls_w28_{num_frame_split}frame_s{start_f}i5.pt'))
