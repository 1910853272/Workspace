import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
import sys
sys.path.append('.')  # 将当前路径添加到系统路径，以便导入模块

# 定义DvsTFDataset类，用于加载DVS的Tensor数据集
class DvsTFDataset(Dataset):
    def __init__(self, path) -> None:
        super(DvsTFDataset, self).__init__()
        # 从指定路径加载数据和标签
        self.data, self.label = torch.load(path)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 返回指定索引的数据和标签
        return self.data[idx], self.label[idx]

# 从给定的文件名列表中读取数据文件
def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

# 从HDF5文件中加载数据和标签
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# 从指定文件加载数据和标签
def loadDataFile(filename):
    return load_h5(filename)

# 从HDF5文件中加载数据、标签和分割信息
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# 从指定文件加载数据、标签和分割信息
def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

# 定义DvsDataset类，用于加载DVS数据集
class DvsDataset(Dataset):
    def __init__(self, DATADIR, train, num_points=1024, use_raw=True):
        super(DvsDataset, self).__init__()
        self.num_points = num_points  # 每个样本的点数
        self.use_raw = use_raw  # 是否使用原始数据

        # 确定数据集目录并加载文件
        if self.use_raw:
            # 根据训练/测试选择数据目录
            self.dataset_dir = os.path.join(
                DATADIR, "train") if train else os.path.join(DATADIR, "test")
            files = os.listdir(self.dataset_dir)  # 获取目录下的所有文件
            print("processing dataset:{} ".format(self.dataset_dir))
        else:
            # 根据训练/测试加载文件列表
            files = getDataFiles(os.path.join(DATADIR, 'train_files.txt')) if train else getDataFiles(
                os.path.join(DATADIR, 'test_files.txt'))
            print("processing dataset:{} ".format(DATADIR))

        # 初始化数据和标签列表
        self.data, self.label = [], []
        if self.use_raw:
            # 如果使用原始数据，从每个文件中读取数据和标签
            for f in files:
                with open(os.path.join(self.dataset_dir, f), 'rb') as f:
                    dataset = pickle.load(f)
                self.data += dataset['data']
                self.label += dataset['label'].tolist()
        else:
            # 否则从HDF5文件中读取数据和标签
            for f in files:
                d, l = loadDataFile(os.path.join(DATADIR, f))
                self.data.append(d)
                self.label.append(l)
            # 将数据和标签连接起来
            self.data = np.concatenate(self.data, axis=0).squeeze()
            self.label = np.concatenate(self.label, axis=0).squeeze()

    def __getitem__(self, index):
        # 获取指定索引的样本数据和标签
        if self.use_raw:
            label = int(self.label[index])  # 标签为整数
            events = self.data[index]  # 获取事件数据
            nr_events = events.shape[0]  # 获取事件的数量
            idx = np.arange(nr_events)  # 生成索引数组
            np.random.shuffle(idx)  # 随机打乱索引
            idx = idx[0: self.num_points]  # 选择前num_points个索引
            events = events[idx, ...]  # 根据索引获取事件

            # 对事件进行归一化处理
            events_normed = np.zeros_like(events, dtype=np.float32)
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            events_normed[:, 1] = x / 127  # 归一化x坐标
            events_normed[:, 2] = y / 127  # 归一化y坐标
            t = t - np.min(t)  # 将时间减去最小值
            t = t / np.max(t)  # 归一化时间
            events_normed[:, 0] = t

            return events_normed, label  # 返回归一化后的事件和标签
        else:
            # 如果不使用原始数据，直接返回数据和标签
            return self.data[index], self.label[index]

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

# 主函数，用于测试数据集的加载
if __name__ == '__main__':
    DATADIR = 'data/DVS_C10_TS1_1024'  # 数据集目录
    tr = DvsDataset(DATADIR, train=True)  # 加载训练集
    length = len(tr)  # 获取数据集大小
    print(length)  # 打印数据集大小