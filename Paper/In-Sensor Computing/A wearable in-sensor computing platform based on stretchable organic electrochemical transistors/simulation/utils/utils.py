from PIL.Image import new  # 导入PIL库用于图像操作
from numpy.core.fromnumeric import squeeze  # 用于压缩数组维度
import pandas  # 导入pandas库用于数据处理
import pandas as pd  # 导入pandas库
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数接口
import os  # 导入操作系统相关模块
import numpy as np  # 导入numpy库用于数组操作
import matplotlib.pyplot as plt  # 导入matplotlib用于绘制图形
import cv2  # 导入OpenCV库用于图像处理

# 数据处理函数，读取Excel文件并根据特定规则提取数据
def oect_data_proc(path, device_test_cnt, num_pulse=5, device_read_times=None):
    # 读取Excel文件并设置'pulse'列为字符串格式
    device_excel = pd.read_excel(path, converters={'pulse': str})

    # 定义时间节点
    device_read_time_list = ['10s', '10.5s', '11s', '11.5s', '12s']
    if device_read_times == None:
        cnt = 0  # 若未指定读取时间，默认为0
    else:
        cnt = device_read_time_list.index(device_read_times)  # 获取读取时间的索引

    # 每次读取的行数，依据num_pulse来计算
    num_rows = 2 ** num_pulse
    # 根据指定的读取时间区间，提取相应的数据
    device_data = device_excel.iloc[cnt * (num_rows + 1): cnt * (num_rows + 1) + num_rows, 0: device_test_cnt + 1]

    # 将'pulse'列作为DataFrame的索引，并删除'pulse'列
    device_data.index = device_data['pulse']
    del device_data['pulse']

    return device_data  # 返回处理后的设备数据

# 二值化数据函数，基于阈值将数据转化为0或1
def binarize_dataset(data, threshold):
    data = torch.where(data > threshold * data.max(), 1, 0)  # 如果数据大于阈值的最大值，则设置为1，否则设置为0
    return data  # 返回二值化后的数据

# 重新排列数据函数，按照给定的num_pulse将数据重新组织
def reshape(data, num_pulse):
    num_data, h, w = data.shape  # 获取数据的维度
    new_data = []
    for i in range(int(w / num_pulse)):  # 根据num_pulse的值将宽度w划分成若干组
        new_data.append(data[:, :, i * num_pulse: (i+1) * num_pulse])  # 将每一组数据提取并添加到new_data中

    new_data = torch.cat(new_data, dim=1)  # 将new_data沿着第二个维度（宽度）拼接
    return new_data  # 返回重新排列的数据

# 自定义数据集类
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 num_pulse,
                 crop=False,
                 transform=None,
                 sampling=0,
                 ori_img=False):
        '''
        ori_img: 如果返回原始MNIST图像及其裁剪和缩放后的图像
        '''
        super(SimpleDataset, self).__init__()

        self.get_ori_img = ori_img  # 设置是否获取原始图像

        if type(path) is str:
            self.data, self.label = torch.load(path)  # 加载数据和标签
        elif type(path) is tuple:
            self.data, self.label = path[0], path[1]  # 如果path是元组，直接分配
        else:
            print('wrong path type')  # 如果路径类型不对，报错

        self.ori_img = self.data  # 原始图像数据

        if crop:
            # 对图像进行裁剪操作
            self.data = self.data[:, 5: 25, 5: 25]
        if sampling != 0:
            self.data = self.data.unsqueeze(dim=1)  # 增加一个维度，进行插值操作
            self.data = F.interpolate(self.data, size=(sampling, sampling))  # 重新调整图像大小
            self.data = self.data.squeeze()  # 去除增加的维度

        self.new_img = self.data  # 更新后的图像

        num_data = self.data.shape[0]  # 数据数量
        img_h, img_w = self.data.shape[1], self.data.shape[2]  # 图像的高度和宽度
        num_pixel = img_h * img_w  # 图像的像素数

        # 如果path是字符串类型，进行二值化和重新排列
        if type(path) is str:
            self.data = binarize_dataset(self.data, threshold=0.25)  # 二值化数据
            self.data = reshape(self.data, num_pulse)  # 重新排列数据
            self.data = torch.transpose(self.data, dim0=1, dim1=2)  # 交换维度

        else:
            self.data = torch.squeeze(self.data)  # 去除不必要的维度
            self.label = torch.squeeze(self.label)  # 去除标签中的不必要维度

        if transform is not None:
            self.transform = transform  # 如果有数据变换，则使用变换

    # 获取数据集中的一个样本
    def __getitem__(self, index):
        target = self.label[index]  # 获取对应的标签
        img = self.data[index]  # 获取对应的图像数据
        if self.get_ori_img:
            # 如果需要原始图像，则返回原始图像及其裁剪后的图像
            return img, self.ori_img[index], self.new_img[index], target
        else:
            # 只返回图像数据和标签
            return img, target

    # 返回数据集的大小
    def __len__(self):
        return self.data.shape[0]

# RC特征提取函数，使用设备输出提取输入数据的特征
def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    使用设备输出提取特征
    :param data: 输入数据
    :param device_data: 设备输出数据，DataFrame格式
    :param device_tested_number: 设备的测试数量
    :param num_pulse: 每个脉冲的数量
    :param padding: 是否对数据进行补充
    :return: 提取的特征
    '''
    device_outputs = []  # 存储设备输出
    img_width = data.shape[-1]  # 获取图像的宽度
    for i in range(img_width):  # 遍历图像的每一列
        rand_ind = np.random.randint(1, device_tested_number + 1)  # 随机选择设备输出的索引
        # 将数据转换为二进制字符串
        if len(data.shape) == 3:
            ind = [str(idx) for idx in data[0, :, i].numpy()]
        elif len(data.shape) == 2:
            ind = [str(idx) for idx in data[:, i].numpy()]
        ind = ''.join(ind)  # 将二进制位连接成一个字符串

        # 根据不同的设备测试数量，从设备数据中提取相应的输出
        if num_pulse == 4 and padding:
            ind = '1' + ind
            output = device_data.loc[ind, rand_ind]
        elif device_tested_number in [2, 4, 5]:
            output = device_data.loc[ind, rand_ind]
        elif device_tested_number == 1:
            output = device_data.loc[ind]
        device_outputs.append(output)  # 将输出加入到设备输出列表中

    device_outputs = torch.unsqueeze(torch.tensor(device_outputs, dtype=torch.float), dim=0)  # 将输出转换为张量
    return device_outputs  # 返回提取的特征

# 批量RC特征提取函数
def batch_rc_feat_extract(data,
                          device_output,
                          device_tested_number,
                          num_pulse,
                          batch_size):
    features = []  # 存储特征
    for batch in range(batch_size):  # 遍历批次
        single_data = data[batch]  # 获取单个数据
        feature = rc_feature_extraction(single_data, device_output, device_tested_number, num_pulse)  # 提取特征
        features.append(feature)  # 将特征加入到特征列表中

    features = torch.cat(features, dim=0)  # 将所有特征沿着第一个维度拼接
    return features  # 返回拼接后的特征

# 用于演示的图像管理类
class ImagesForDemo():
    def __init__(self, path) -> None:
        self.ori_images = []  # 存储原始图像
        self.new_images = []  # 存储新图像
        self.reshaped_data = []  # 存储重排后的数据
        self.probabilites = []  # 存储概率
        self.targets = []  # 存储标签
        self.path = path  # 保存路径
        if os.path.exists(path) is not True:
            os.mkdir(path)  # 如果路径不存在，则创建路径

    # 更新图像数据
    def update_images(self, data, img, new_img, target, output):
        p = F.softmax(output, dim=1).max()  # 计算概率
        if output.argmax(dim=-1) == target:  # 如果预测结果与真实标签匹配
            if target not in self.targets:  # 如果标签不在已有标签中
                self.ori_images.append(img)  # 添加原始图像
                self.reshaped_data.append(data)  # 添加重排后的数据
                self.new_images.append(new_img)  # 添加新图像
                self.targets.append(target)  # 添加标签
                self.probabilites.append(p)  # 添加概率
            else:
                idx = self.targets.index(target)  # 查找标签的位置
                if p > self.probabilites[idx]:  # 如果新图像的概率更高
                    self.probabilites[idx] = p  # 更新概率
                    self.ori_images[idx] = img  # 更新原始图像
                    self.new_images[idx] = new_img  # 更新新图像
                    self.reshaped_data[idx] = data  # 更新重排后的数据

    # 保存图像
    def save_images(self):
        for i, target in enumerate(self.targets):  # 遍历所有标签
            target = target.tolist()[0]  # 获取标签
            cls_path = os.path.join(self.path, str(target))  # 创建类路径
            if os.path.exists(cls_path) is not True:
                os.mkdir(cls_path)  # 如果类路径不存在，则创建路径
            image_name = f'image_confidence_{int(self.probabilites[i] * 10000): d}.jpg'  # 根据概率命名图像
            image_name = os.path.join(cls_path, image_name)  # 完整的图像路径
            cropped_img_name = os.path.join(cls_path, 'cropped_image')  # 裁剪图像路径
            pulse_name = os.path.join(cls_path, 'pulses')  # 脉冲图像路径

            cv2.imwrite(image_name, self.ori_images[i].squeeze().numpy())  # 保存原始图像
            plt.figure()  # 创建图形
            plt.imshow(self.reshaped_data[i].squeeze())  # 显示重排后的图像
            plt.savefig(pulse_name)  # 保存脉冲图像
            plt.close()  # 关闭图形

            # 保存裁剪图像
            plt.figure()
            plt.imshow(self.new_images[i].squeeze())
            plt.savefig(cropped_img_name)
            plt.close()
