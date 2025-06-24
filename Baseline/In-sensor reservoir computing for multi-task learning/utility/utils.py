<<<<<<< HEAD
# 导入相关的库，包括PyTorch、Pandas、NumPy等用于深度学习、数据处理和图像处理
from numpy.core.fromnumeric import size, squeeze
import pandas
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.datasets.mnist import EMNIST, MNIST, FashionMNIST
import argparse
import copy

import torch
from torch.utils.data import Dataset

# ann_baseline 报错 AttributeError: module 'utility.utils' has no attribute 'SimpleDataset'

# 将二进制数据转换为十进制表示
def bin2dec(b, bits):
    # 创建掩码，按位权重从大到小排列
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = mask.unsqueeze(-1)
    # 根据掩码计算出对应的十进制值
    return torch.sum(b * mask, dim=1)



# 处理设备数据（4月份的数据处理方式）
def oect_data_proc_04(path, device_tested_number):
    '''
    for April data processing
    处理4月份的设备测试数据
    '''
    # 读取Excel文件，pulse列使用字符串类型
    device_excel = pandas.read_excel(path, converters={'pulse': str})

    # 获取pulse列
    device_excel['pulse']

    # 读取前 device_tested_number 列数据并设置索引
    device_data = device_excel.iloc[:, 1:device_tested_number+1]
    device_data.iloc[30] = 0  # 将第30行设置为0
    ind = device_excel['pulse']
    # 处理字符串中的特殊字符
    ind = [str(i).split('‘')[-1].split('’')[-1].split('\'')[-1] for i in ind]
    device_data.index = ind

    return device_data

# 处理设备数据（5月7日的数据处理方式）
def oect_data_proc(path, device_test_cnt, num_pulse=5, device_read_times=None):
    '''
    for 0507 data processing
    '''
    # 读取Excel文件，pulse列使用字符串类型
    device_excel = pd.read_excel(path, converters={'pulse': str})

    # 定义一个固定的读取时间点列表
    device_read_time_list = ['10s', '10.5s', '11s', '11.5s', '12s']
    if device_read_times == None:
        cnt = 0
    else:
        cnt = device_read_time_list.index(device_read_times)

    # 根据脉冲数确定数据行数
    num_rows = 2 ** num_pulse
    # 根据时间段读取相关数据
    device_data = device_excel.iloc[cnt * (num_rows + 1): cnt * (num_rows + 1) + num_rows, 0: device_test_cnt + 1]
    del device_data['pulse']  # 删除pulse列

    return device_data

# 标准化的数据处理函数
def oect_data_proc_std(path, device_test_cnt, num_pulse=5):
    '''
    standard processing function
    标准化数据处理函数
    '''
    device_excel = pd.read_excel(path, converters={'pulse': str})

    # 删除pulse列
    del device_excel['pulse']

    return device_excel




# 利用设备数据提取图像特征
def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    use device to extract feature (randomly select a experimental output value corresponding to the input binary digits)
    使用设备数据提取特征（随机选择一个实验输出值与输入的二进制数据对应）
    '''
    img_width = data.shape[-1]
    device_outputs = torch.empty((1, img_width))
    for i in range(img_width):
        # 图像数据的二进制索引
        ind = [num ** (5 - idx) for idx, num in enumerate(data[:, i].numpy())]
        ind = int(np.sum(ind))
        if num_pulse == 4 and padding:
            ind += 16
        if device_tested_number > 1:
            # 随机选择设备输出
            rand_ind = np.random.randint(1, device_tested_number)
            output = device_data[ind, rand_ind]
        else:
            output = device_data[ind, 0]
        device_outputs[0, i] = output.item()  # 将输出结果存入device_outputs
    return device_outputs

# 批量提取特征
def batch_rc_feat_extract(data,
                          device_output,
                          device_tested_number,
                          num_pulse,
                          batch_size):
    features = []
    # 遍历批次
    for batch in range(batch_size):
        single_data = data[batch]
        # 提取单个批次的数据特征
        feature = rc_feature_extraction(single_data,
                                        device_output,
                                        device_tested_number,
                                        num_pulse)
        features.append(feature)
    features = torch.cat(features, dim=0)  # 合并所有批次的特征
    return features

# 批量提取特征，尚未完成
def batch_rc_feat_extract_in_construction(data,
                          device_output,
                          device_tested_number,
                          start_idx, # start idx for device tested number
                          num_pulse,
                          batch_size):
    '''
    data: a batch of data. shape: (batch_size, 5, 28* 28) for dvs image
          (batch_size, 5, 140) for old mnist data (to check)
    output: a batch of features. shape: (batch_size, 1, 28, 28)
    '''
    # 将二进制数据转换为十进制
    data_seq = bin2dec(data, num_pulse).numpy().astype(int)

    # 随机生成序列
    data_random_seq = np.random.randint(1, 2, data_seq.shape)

    # 提取设备输出的特征
    feat = device_output[data_seq, data_random_seq]
    del data, data_seq, data_random_seq
    return torch.tensor(feat)

# 写入日志
def write_log(save_dir_name, log):
    if type(log) != str:
        log = str(log)
    log_file_name = os.path.join(save_dir_name, 'log.txt')
    with open(log_file_name, 'a') as f:
        f.writelines(log)


# 合并上行和下行的条件
def conds_combine(conds_up, conds_down):
    conds = np.concatenate((conds_up, conds_down), axis=0)
    conds = np.sort(conds, axis=0)  # 按列排序
    return conds

# 将权重映射到条件值
def w2c_mapping(weight, conds, weights_limit):
    weight_clipped = torch.where(weight > weights_limit, weights_limit, weight)
    weight_clipped = torch.where(weight_clipped < -weights_limit, -weights_limit, weight_clipped)
    # 线性映射权重值到设备条件
    a = (conds.max() - conds.min()) / (weight_clipped.max() - weight_clipped.min()) 
    b = conds.min() - weight_clipped.min() * a
    return a.item(), b.item()

# 寻找最接近的值
def find_nearest(value_array, query_mat):
    # 为每个查询点扩展value_array以匹配形状
    query_mat_stack = np.tile(query_mat, [value_array.shape[0], 1, 1]).transpose(1, 2, 0)

    # 计算与每个value的差异，并找到差异最小的索引
    differences = query_mat_stack - value_array
    indices = np.argmin(np.abs(differences), axis=-1)
    # 返回与查询值最接近的value_array中的值
    values = value_array[indices]
    return values

# 将权重值转换为设备条件
def weight2cond(weight, conds, a, b):
    cond = a * weight + b
    cond = find_nearest(conds, cond)
    return cond

# 将设备条件转换回权重值
def cond2weight(cond, a, b):
    weight = (cond - b) / a
    return weight

# 对模型参数进行量化
def model_quantize(model, conds, weights_limit_ratio=0.9, plot_weight=False, save_dir_name=''):
    for name, weight in model.named_parameters():
        if 'weight' not in name:
            continue
        # 根据权重限制比例确定限制值
        weights_limit, _ = torch.sort(weight.data.abs().flatten(), descending=False)
        weights_limit =  weights_limit[int(weights_limit_ratio * len(weights_limit))]
        # 映射权重到设备条件
        a, b = w2c_mapping(weight.data, conds, weights_limit)
        cond_data = weight2cond(weight.data, conds, a, b)
        weight_data = cond2weight(cond_data, a, b)
        weight.data = torch.tensor(weight_data, dtype=torch.float32)

    return model

# 梯度映射函数（未完成）
def gradient_mapping(weight, gradient, up_table, down_tabel, min_cond, max_cond, weight_upper_limit):
    sign_weight = torch.sign(weight)
    sign_gradient = torch.sign(gradient)
    gradient = torch.where(sign_gradient)
    pass

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()

    # 选择数据集，支持MNIST、EMNIST和FashionMNIST
    parser.add_argument('--dataset', type=str, default='FMNIST', choices=['MNIST', 'EMNIST', 'FMNIST'], help='choose dataset')
    parser.add_argument('--split', type=str, default='letters', choices=['letters', 'bymerge', 'byclass'], help='emnist split method')

    '''DEVICE FILE'''
    parser.add_argument('--device_file', type=str, default='p_NDI_05s', help='device file')

    # 选择设备（CPU或GPU）
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'], help='cuda device')

    # 设置脉冲数和其他参数
    parser.add_argument('--num_pulse', type=int, default=5, help='the number of pulse in one sequence. (For '
                        'train with feature, num_pulse should be 1)')
    parser.add_argument('--crop', type=str, default=False, help='crop the images')
    parser.add_argument('--sampling', type=int, default=0, help='image downsampling')
    parser.add_argument('--bin_threshold', type=float, default=0.25, help='binarization thershold')
    parser.add_argument('--device_test_num', type=int, default=1)

    # 是否使用数字作为输出
    parser.add_argument('--digital', type=bool, default=False, help='use digits as reservoir output')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_step_size', type=int, default=70, help='learning rate step')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='learning rate gamma')

    # 设置模式为模拟或真实神经网络
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'], help='sim: our simulate network, real: real ann network')
    parser.add_argument('--a_w2c', type=float, default=10)
    parser.add_argument('--bias_w2c', type=float, default=0.1)
    args = parser.parse_args()
    return args
=======
# 导入相关的库，包括PyTorch、Pandas、NumPy等用于深度学习、数据处理和图像处理
from numpy.core.fromnumeric import size, squeeze
import pandas
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.datasets.mnist import EMNIST, MNIST, FashionMNIST
import argparse
import copy

import torch
from torch.utils.data import Dataset

# ann_baseline 报错 AttributeError: module 'utility.utils' has no attribute 'SimpleDataset'

# 将二进制数据转换为十进制表示
def bin2dec(b, bits):
    # 创建掩码，按位权重从大到小排列
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = mask.unsqueeze(-1)
    # 根据掩码计算出对应的十进制值
    return torch.sum(b * mask, dim=1)



# 处理设备数据（4月份的数据处理方式）
def oect_data_proc_04(path, device_tested_number):
    '''
    for April data processing
    处理4月份的设备测试数据
    '''
    # 读取Excel文件，pulse列使用字符串类型
    device_excel = pandas.read_excel(path, converters={'pulse': str})

    # 获取pulse列
    device_excel['pulse']

    # 读取前 device_tested_number 列数据并设置索引
    device_data = device_excel.iloc[:, 1:device_tested_number+1]
    device_data.iloc[30] = 0  # 将第30行设置为0
    ind = device_excel['pulse']
    # 处理字符串中的特殊字符
    ind = [str(i).split('‘')[-1].split('’')[-1].split('\'')[-1] for i in ind]
    device_data.index = ind

    return device_data

# 处理设备数据（5月7日的数据处理方式）
def oect_data_proc(path, device_test_cnt, num_pulse=5, device_read_times=None):
    '''
    for 0507 data processing
    '''
    # 读取Excel文件，pulse列使用字符串类型
    device_excel = pd.read_excel(path, converters={'pulse': str})

    # 定义一个固定的读取时间点列表
    device_read_time_list = ['10s', '10.5s', '11s', '11.5s', '12s']
    if device_read_times == None:
        cnt = 0
    else:
        cnt = device_read_time_list.index(device_read_times)

    # 根据脉冲数确定数据行数
    num_rows = 2 ** num_pulse
    # 根据时间段读取相关数据
    device_data = device_excel.iloc[cnt * (num_rows + 1): cnt * (num_rows + 1) + num_rows, 0: device_test_cnt + 1]
    del device_data['pulse']  # 删除pulse列

    return device_data

# 标准化的数据处理函数
def oect_data_proc_std(path, device_test_cnt, num_pulse=5):
    '''
    standard processing function
    标准化数据处理函数
    '''
    device_excel = pd.read_excel(path, converters={'pulse': str})

    # 删除pulse列
    del device_excel['pulse']

    return device_excel




# 利用设备数据提取图像特征
def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    use device to extract feature (randomly select a experimental output value corresponding to the input binary digits)
    使用设备数据提取特征（随机选择一个实验输出值与输入的二进制数据对应）
    '''
    img_width = data.shape[-1]
    device_outputs = torch.empty((1, img_width))
    for i in range(img_width):
        # 图像数据的二进制索引
        ind = [num ** (5 - idx) for idx, num in enumerate(data[:, i].numpy())]
        ind = int(np.sum(ind))
        if num_pulse == 4 and padding:
            ind += 16
        if device_tested_number > 1:
            # 随机选择设备输出
            rand_ind = np.random.randint(1, device_tested_number)
            output = device_data[ind, rand_ind]
        else:
            output = device_data[ind, 0]
        device_outputs[0, i] = output.item()  # 将输出结果存入device_outputs
    return device_outputs

# 批量提取特征
def batch_rc_feat_extract(data,
                          device_output,
                          device_tested_number,
                          num_pulse,
                          batch_size):
    features = []
    # 遍历批次
    for batch in range(batch_size):
        single_data = data[batch]
        # 提取单个批次的数据特征
        feature = rc_feature_extraction(single_data,
                                        device_output,
                                        device_tested_number,
                                        num_pulse)
        features.append(feature)
    features = torch.cat(features, dim=0)  # 合并所有批次的特征
    return features

# 批量提取特征，尚未完成
def batch_rc_feat_extract_in_construction(data,
                          device_output,
                          device_tested_number,
                          start_idx, # start idx for device tested number
                          num_pulse,
                          batch_size):
    '''
    data: a batch of data. shape: (batch_size, 5, 28* 28) for dvs image
          (batch_size, 5, 140) for old mnist data (to check)
    output: a batch of features. shape: (batch_size, 1, 28, 28)
    '''
    # 将二进制数据转换为十进制
    data_seq = bin2dec(data, num_pulse).numpy().astype(int)

    # 随机生成序列
    data_random_seq = np.random.randint(1, 2, data_seq.shape)

    # 提取设备输出的特征
    feat = device_output[data_seq, data_random_seq]
    del data, data_seq, data_random_seq
    return torch.tensor(feat)

# 写入日志
def write_log(save_dir_name, log):
    if type(log) != str:
        log = str(log)
    log_file_name = os.path.join(save_dir_name, 'log.txt')
    with open(log_file_name, 'a') as f:
        f.writelines(log)


# 合并上行和下行的条件
def conds_combine(conds_up, conds_down):
    conds = np.concatenate((conds_up, conds_down), axis=0)
    conds = np.sort(conds, axis=0)  # 按列排序
    return conds

# 将权重映射到条件值
def w2c_mapping(weight, conds, weights_limit):
    weight_clipped = torch.where(weight > weights_limit, weights_limit, weight)
    weight_clipped = torch.where(weight_clipped < -weights_limit, -weights_limit, weight_clipped)
    # 线性映射权重值到设备条件
    a = (conds.max() - conds.min()) / (weight_clipped.max() - weight_clipped.min()) 
    b = conds.min() - weight_clipped.min() * a
    return a.item(), b.item()

# 寻找最接近的值
def find_nearest(value_array, query_mat):
    # 为每个查询点扩展value_array以匹配形状
    query_mat_stack = np.tile(query_mat, [value_array.shape[0], 1, 1]).transpose(1, 2, 0)

    # 计算与每个value的差异，并找到差异最小的索引
    differences = query_mat_stack - value_array
    indices = np.argmin(np.abs(differences), axis=-1)
    # 返回与查询值最接近的value_array中的值
    values = value_array[indices]
    return values

# 将权重值转换为设备条件
def weight2cond(weight, conds, a, b):
    cond = a * weight + b
    cond = find_nearest(conds, cond)
    return cond

# 将设备条件转换回权重值
def cond2weight(cond, a, b):
    weight = (cond - b) / a
    return weight

# 对模型参数进行量化
def model_quantize(model, conds, weights_limit_ratio=0.9, plot_weight=False, save_dir_name=''):
    for name, weight in model.named_parameters():
        if 'weight' not in name:
            continue
        # 根据权重限制比例确定限制值
        weights_limit, _ = torch.sort(weight.data.abs().flatten(), descending=False)
        weights_limit =  weights_limit[int(weights_limit_ratio * len(weights_limit))]
        # 映射权重到设备条件
        a, b = w2c_mapping(weight.data, conds, weights_limit)
        cond_data = weight2cond(weight.data, conds, a, b)
        weight_data = cond2weight(cond_data, a, b)
        weight.data = torch.tensor(weight_data, dtype=torch.float32)

    return model

# 梯度映射函数（未完成）
def gradient_mapping(weight, gradient, up_table, down_tabel, min_cond, max_cond, weight_upper_limit):
    sign_weight = torch.sign(weight)
    sign_gradient = torch.sign(gradient)
    gradient = torch.where(sign_gradient)
    pass

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()

    # 选择数据集，支持MNIST、EMNIST和FashionMNIST
    parser.add_argument('--dataset', type=str, default='FMNIST', choices=['MNIST', 'EMNIST', 'FMNIST'], help='choose dataset')
    parser.add_argument('--split', type=str, default='letters', choices=['letters', 'bymerge', 'byclass'], help='emnist split method')

    '''DEVICE FILE'''
    parser.add_argument('--device_file', type=str, default='p_NDI_05s', help='device file')

    # 选择设备（CPU或GPU）
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'], help='cuda device')

    # 设置脉冲数和其他参数
    parser.add_argument('--num_pulse', type=int, default=5, help='the number of pulse in one sequence. (For '
                        'train with feature, num_pulse should be 1)')
    parser.add_argument('--crop', type=str, default=False, help='crop the images')
    parser.add_argument('--sampling', type=int, default=0, help='image downsampling')
    parser.add_argument('--bin_threshold', type=float, default=0.25, help='binarization thershold')
    parser.add_argument('--device_test_num', type=int, default=1)

    # 是否使用数字作为输出
    parser.add_argument('--digital', type=bool, default=False, help='use digits as reservoir output')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_step_size', type=int, default=70, help='learning rate step')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='learning rate gamma')

    # 设置模式为模拟或真实神经网络
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'], help='sim: our simulate network, real: real ann network')
    parser.add_argument('--a_w2c', type=float, default=10)
    parser.add_argument('--bias_w2c', type=float, default=0.1)
    args = parser.parse_args()
    return args
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
