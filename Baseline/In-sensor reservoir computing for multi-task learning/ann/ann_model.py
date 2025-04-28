import torch
import torch.nn as nn
import numpy as np
import sys


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

# 定义一个自定义的模型类，继承自torch.nn.Module
class model(nn.Module):
    """
    自定义模型类，用于实现特定的神经网络模型结构和功能。
    
    参数:
    - hid_dim: 隐藏层的维度
    - num_class: 分类的类别数量
    - batchsize: 批处理的大小
    - num_layer: 网络的层数
    - conds_up: 上限条件张量
    - conds_down: 下限条件张量
    - a_w2c: 权重到条件的转换系数
    - bias_w2c: 权重到条件的偏置
    - config: 配置字典，包含模型的特定配置参数
    - device: 模型运行的设备,默认为cpu
    """
    def __init__(self,
                 hid_dim,
                 num_class,
                 batchsize,
                 num_layer,
                 conds_up,
                 conds_down,
                 a_w2c,
                 bias_w2c,
                 config,
                 device=torch.device('cpu')):
        super(model, self).__init__()
        self.hid_dim = hid_dim
        self.batchsize = batchsize
        self.num_layer = num_layer
        self.num_class = num_class
        
        # 定义输出层，使用全连接层（线性变换）将隐藏层的输出转换为类别数量的输出
        self.fc_out = nn.Linear(int(self.hid_dim), num_class)
        
        # 处理条件张量，确保其为torch.Tensor类型，并转移到指定设备
        if type(conds_up) != torch.Tensor:
            conds_up = torch.Tensor(conds_up)
            conds_down = torch.Tensor(conds_down)
        conds_up = conds_up.unsqueeze(0)  # 在第一维增加一个维度
        conds_down = conds_down.unsqueeze(0)  # 在第一维增加一个维度
        self.conds_up = conds_up.to(device)  # 转移到指定设备
        self.conds_down = conds_down.to(device)  # 转移到指定设备
        
        # 计算条件张量的最大和最小值，用于后续的条件限制
        self.conds_max = max(conds_up.max(), conds_down.max())
        self.conds_min = min(conds_up.min(), conds_down.min())
        
        # 初始化是否处理多周期条件的标志
        self.multi_cycles = False
        # 根据条件张量的维度判断是否处理多周期条件
        if len(self.conds_up.shape) > 1:
            self.multi_cycles = True
            # 如果是多周期条件，则保存所有周期的条件张量
            self.conds_up_all = self.conds_up
            self.conds_down_all = self.conds_down
            self.num_cycles = self.conds_up.shape[0]  # 周期的数量
            self.num_pulse = self.conds_up.shape[1]  # 每个周期的脉冲数量
        else:
            self.num_cycles = 1
            self.num_pulse = self.conds_up.shape[1]
        
        # 设置权重到条件的转换系数和偏置，可以根据配置进行调整
        self.a_w2c = a_w2c
        self.bias_w2c = bias_w2c
        # 从配置中读取具体的转换系数和偏置值
        self.a_w2c = config['a_w2c']
        self.bias_w2c = config['bias_w2c']
        
        # 设置梯度到脉冲的转换系数
        self.a_grad = torch.tensor(100)
        
        # 计算条件的上下限，用于权重到条件的转换限制
        self.max_cond = torch.max(self.conds_up.max(), self.conds_down.max())
        self.min_cond = torch.min(self.conds_up.min(), self.conds_down.min())
    
    # 根据权重和梯度计算新的导通条件
    # weight2cond函数用于根据当前的权重和梯度，计算出新的导通条件，以此来更新神经网络中的导电材料的导通状态。
    # 该函数主要处理的是在多周期情况下，如何根据当前的权重和梯度来确定导通条件的上下限，以及如何根据这些条件来量化权重。
    def weight2cond(self, weight, grad):
        # 如果启用多周期，则随机选择一个周期的导通条件
        if self.multi_cycles == True:
            # 在所有周期中随机选择一个上行周期和一个下行周期
            up_cycle_idx = np.random.randint(self.num_cycles)
            down_cycle_idx = np.random.randint(self.num_cycles)

            # 更新上行和下行导通条件为选中的周期的条件
            self.conds_up = self.conds_up_all[up_cycle_idx, :]
            self.conds_down = self.conds_down_all[down_cycle_idx, :]

        # 计算基础导通条件
        cond = weight * self.a_w2c + self.bias_w2c

        # 确定梯度方向
        direction = torch.sign(grad)  # 梯度方向
        # 确定梯度的正负区域
        pos_mat = torch.where(direction >= 0, 1, 0)  # 梯度正负区域
        # 判断导通条件是否超出上下限
        up_overflow, down_overflow = cond > self.max_cond, cond < self.min_cond

        # 更新导通条件，超出上限的设置为上限，超出下限的设置为下限
        cond_new = torch.where(up_overflow, self.max_cond.to(torch.float), cond)
        cond_new = torch.where(down_overflow, self.min_cond.to(torch.float), cond_new)

        # 对导通条件进行量化处理
        ori_shape = cond_new.shape  # 保存原始形状以便后续还原
        cond_flatten = cond_new.reshape(-1)  # 将张量展平为一维
        pos_mat_flatten = pos_mat.reshape(-1)
        up_overflow_flatten, down_overflow_flatten = up_overflow.reshape(-1), down_overflow.reshape(-1)
        indices_flatten = torch.zeros_like(cond_flatten, dtype=torch.int)  # 初始化索引张量
        # 根据导通条件的位置和方向，确定其对应的量化索引和值
        for i, (c, pos_sign, up_of, down_of) in enumerate(zip(cond_flatten, pos_mat_flatten, up_overflow_flatten, down_overflow_flatten)):
            if up_of:
                indices_flatten[i] = self.num_pulse - 1  # 如果超出上限，设置为最大脉冲数
                cond_flatten[i] = self.conds_up[-1]
            elif down_of:
                if pos_sign: indices_flatten[i] = 0
                else: indices_flatten[i] = self.num_pulse - 1
                cond_flatten[i] = self.conds_up[0]
            else:
                # 根据导通条件的方向，寻找最近的导通条件值
                if pos_sign:
                    idx, value = find_nearest(array=self.conds_up, key=c)
                else:
                    idx, value = find_nearest(array=self.conds_down, key=c)
                indices_flatten[i] = idx
                cond_flatten[i] = value

        # 将量化后的导通条件恢复到原来的形状
        indices = indices_flatten.reshape(ori_shape).to(torch.int64)
        cond_new = cond_flatten.reshape(ori_shape)
        # 返回量化后的导通条件索引、新的导通条件和梯度方向矩阵
        return indices, cond_new, pos_mat

    # 该方法用于根据条件计算权重的调整值
    def cond2weight(self, cond):
        """
        根据给定的条件，计算权重的调整值。
        
        参数:
        cond: 条件值，用于计算权重调整。
        
        返回:
        权重调整值，用于更新神经网络的权重。
        """
        return (cond - self.bias_w2c) / self.a_w2c

    # 该方法用于进行梯度更新，并计算权重的新值。
    def gradient_update(self, gradient, weight):
        """
        根据梯度和当前权重，更新权重值。
        
        参数:
        gradient: 梯度值，用于计算权重更新。
        weight: 当前的权重值。
        
        返回:
        更新后的权重值、更新后的条件值、脉冲数、更新后的索引值。
        """
        # 将梯度映射到脉冲数
        num_pulse = gradient * self.a_grad
        num_pulse = num_pulse.to(dtype=torch.int64)

        # 根据权重和梯度计算条件值和符号
        indices, cond, pos_cycle_sign = self.weight2cond(weight, gradient)

        # 更新索引值
        updated_idx = indices - num_pulse.abs()
        updated_idx = updated_idx.to(torch.int64)

        # 确保索引值在有效范围内
        updated_idx = torch.where(updated_idx >= self.num_pulse, self.num_pulse - 1, updated_idx)
        updated_idx = torch.where(updated_idx < 0, 0, updated_idx)

        # 根据符号更新条件值
        updated_cond_up = torch.where(pos_cycle_sign == 1, self.conds_up[updated_idx], torch.tensor(0.).to(torch.float))
        updated_cond_down = torch.where(pos_cycle_sign == 1, torch.tensor(0.).to(torch.float), self.conds_down[updated_idx])
        updated_cond = updated_cond_up + updated_cond_down

        # 根据更新后的条件值计算新的权重值
        updated_weight = self.cond2weight(updated_cond).to(torch.float)

        return updated_weight, updated_cond, num_pulse, updated_idx

    # 该方法是网络的前向传播方法，用于处理输入数据并返回结果。
    def forward(self, x):
        """
        网络的前向传播过程。
        
        参数:
        x: 输入数据。
        
        返回:
        处理后的输出数据。
        """
        # 使用批量归一化处理输入数据
        x = (x - x.mean()) / x.std()
        # 通过全连接层处理数据
        x = self.fc_out(x)
        return x