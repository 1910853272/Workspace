import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST, FashionMNIST, EMNIST
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append('C:\\Users\\19108\Desktop\Project\Memristor\Wearable In-Sensor Reservoir Computing')
from utility.utils import binarize_dataset, single_fig_show, reshape
import copy

# 自定义数据集类，用于加载和预处理简单的图像数据集
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 num_pulse,
                 crop=False,
                 transform=None,
                 sampling=0,
                 ori_img=False,
                 choose_func='train_ete'):
        super(SimpleDataset, self).__init__()

        # 根据选择的功能确定是否对图像进行处理
        if choose_func == 'train_ete' or choose_func == 'save_feat':
            self.image_proc = True
        else:
            self.image_proc = False

        self.get_ori_img = ori_img  # 是否获取原始图像

        # 加载数据和标签
        if type(path) is str:
            self.data, self.targets = torch.load(path)
        elif type(path) is tuple:
            self.data, self.targets = path[0], path[1]
        else:
            print('wrong path type')
        self.ori_img = self.data  # 保存原始图像

        # 根据参数决定是否对图像进行裁剪
        if crop and self.image_proc:
            self.data = self.data[:, 4: 26, 5: 25]
        # 采样处理
        if sampling != 0 and self.image_proc:
            self.data = self.data.unsqueeze(dim=1)  # 增加通道维度
            self.data = F.interpolate(self.data, size=(sampling, sampling))  # 下采样
            self.data = self.data.squeeze()  # 去除多余的维度

        # 可视化采样后的第一个图像
        if len(self.data[0].shape) > 1:
            plt.figure()
            plt.imshow(self.data[0])
            plt.savefig('downsampled_img')

        # 如果数据集来自文件并且需要处理，则进行二值化和数据重塑
        num_data = self.data.shape[0]
        if type(path) is str and self.image_proc:
            self.bin_data = binarize_dataset(self.data, threshold=0.25)  # 二值化

            self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
            self.reshaped_data = reshape(self.bin_data, num_pulse)  # 重塑数据
            self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)  # 转置
            self.data = self.reshaped_data
        else:
            # 否则直接使用原始数据
            self.reshaped_data = torch.squeeze(self.data)
            self.targets = torch.squeeze(self.targets)
        self.transform = transform

    def __getitem__(self, index: int):
        # 获取数据和目标标签
        target = self.targets[index]
        img = self.reshaped_data[index]

        # 如果有变换，应用到图像上
        if self.transform:
            img = self.transform(img)

        # 返回图像及其标签（如果需要，也返回原始图像）
        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def __len__(self):
        return self.data.shape[0]  # 返回数据集的大小

    def get_new_width(self):
        return self.data.shape[-1]  # 获取数据的最后一个维度的宽度

    @property
    def num_class(self):
        return len(set(self.targets.squeeze().tolist()))  # 计算类别数量

    # 可视化特定样本，包括原始图像、二值化图像和脉冲序列
    def visualize_sample(self, save_dir_path, idx=0, cls=0, grid=False):
        if cls:
            idx = torch.nonzero(self.targets == cls)[idx][0]
        ori_sample = self.ori_img[idx]
        bin_sample = self.bin_data[idx]
        pulse_sequences = self.reshaped_data[idx]
        single_fig_show(ori_sample, f'ori_sample_cls{cls}_{idx}', save_dir_path, grid, format='pdf')
        single_fig_show(bin_sample, f'bin_sample_cls{cls}_{idx}', save_dir_path, grid, format='pdf')
        single_fig_show(pulse_sequences, f'pulse_sequences_cls{cls}_{idx}', save_dir_path, grid, grid_width=0.5, format='pdf')

    # 可视化重塑后的脉冲序列
    def visualize_reshaping(self, save_dir_path, idx=0, cls=0, grid=False):
        if cls:
            idx = torch.nonzero(self.targets == cls)[idx][0]
        pulse_sequences = self.reshaped_data[idx]

        len_seg = self.img_h
        num_segment = int(pulse_sequences.shape[1] / len_seg)

        for i in range(num_segment):
            sample = pulse_sequences[:, i * len_seg: (i + 1) * len_seg]
            single_fig_show(sample, f'sample_cls{cls}_{idx}_seg{i}', save_dir_path, grid, format='pdf')

    # 可视化每个类别的原始图像
    def visualize_classes(self, save_dir_path, format='pdf'):
        sample_dict = {}
        for img, target in zip(self.ori_img, self.targets):
            target = target.item()
            if target not in sample_dict.keys():
                sample_dict[target] = img
            if len(sample_dict.keys()) == self.num_class:
                break
        for target, img in sample_dict.items():
            filename = f'class_{target}.jpg'
            filename = os.path.join(save_dir_path, filename)
            cv2.imwrite(filename, img.squeeze().numpy())

# 定义MnistDataset类，继承自MNIST和SimpleDataset
class MnistDataset(MNIST, SimpleDataset):
    def __init__(self,
                 root: str,
                 num_pulse: int,
                 crop=False,
                 sampling=0,
                 mode='sim',
                 ori_img=False,
                 split='letters',
                 **kwargs) -> None:
        super(MnistDataset, self).__init__(root, **kwargs)
        self.get_ori_img = ori_img
        self.ori_img = self.data

        # 根据参数决定是否对图像进行裁剪和采样
        if crop:
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w

        # 二值化处理
        self.bin_data = binarize_dataset(self.data, threshold=0.25)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)
        if mode == 'real':
            self.reshaped_data = torch.squeeze(self.reshaped_data.reshape(num_data, -1)).to(torch.float)

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = self.reshaped_data[index]

        # 如果有变换，应用到图像上
        if self.transform:
            img = self.transform(img)

        # 返回图像及其标签（如果需要，也返回原始图像）
        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def get_new_width(self):
        return self.reshaped_data.shape[-1]  # 获取数据的最后一个维度的宽度

    @property
    def num_class(self):
        return len(set(self.targets.tolist()))  # 计算类别数量

# 其他类似的数据集类EmnistDataset, FmnistDataset, FashionWithSize等也是类似地继承并重载MNIST或FashionMNIST等
# 在这些类中根据不同的数据集需求，进行特定的处理，包括裁剪、二值化、数据重塑、可视化等。

# 这些数据集类可以用来加载、预处理并为训练、评估模型准备数据集，方便我们进一步的机器学习研究与实验。