import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os
from datetime import datetime
import utility.utils as utils
import utility.datasets as datasets
import numpy as np
import utility.train_funcs as train_funcs
from ann import ann_model

# 解析命令行参数
options = utils.parse_args()
num_pulse = options.num_pulse  # 脉冲数量
print(options)

# 时间戳和设备文件名设置
t = datetime.fromtimestamp(time.time())
device_filename = options.device_file + '.xlsx'
save_dir_name = f'{options.device_file}_ete_{datetime.strftime(t, "%m%d%H%M")}'

# 获取目录路径
CODES_DIR = os.path.dirname(os.getcwd())
DATAROOT = 'dataset'  # 数据集路径
DEVICE_DIR = os.path.join(os.getcwd(), 'data')  # 设备数据路径
device_path = os.path.join(DEVICE_DIR, device_filename)  # 设备文件的完整路径
SAVE_PATH = 'C:\\Users\\19108\Desktop\workspace\Baseline\In-sensor reservoir computing for multi-task learning\\rc_sim\log'  # 日志保存路径
save_dir_name = os.path.join(SAVE_PATH, save_dir_name)  # 日志保存目录

# 确保数据集和保存路径存在
for p in [DATAROOT, SAVE_PATH, save_dir_name]:
    if not os.path.exists(p):
        os.mkdir(p)

# 设置一些模型参数
device_tested_number = options.device_test_num  # 测试的设备数量
te_batchsize = 1  # 测试集批量大小

# 训练函数，选择不同的数据集和功能（训练或保存特征）
def train(options, config, dataset, choose_func, train_file='', test_file='', load_pt=False):
    crop = options.crop  # 是否裁剪图像
    sampling = options.sampling  # 下采样参数
    batchsize = options.batch  # 批次大小
    transform = None  # 图像转换设置

    # 加载已经保存的特征文件
    if load_pt:
        tr_dataset = datasets.SimpleDataset(train_file, num_pulse=num_pulse, crop=crop, sampling=sampling, transform=transform, choose_func=choose_func)
        te_dataset = datasets.SimpleDataset(test_file, num_pulse=num_pulse, crop=crop, sampling=sampling, transform=transform, ori_img=True, choose_func=choose_func)
    
    # 如果选择 EMNIST 数据集
    elif dataset == 'EMNIST':
        transform = transforms.Compose([lambda img: transforms.functional.rotate(img, -90), lambda img: transforms.functional.hflip(img)])
        tr_dataset = datasets.EmnistDataset(DATAROOT, num_pulse=num_pulse, crop_class=[1,2,3,4,5,...], crop=crop, sampling=sampling, mode=options.mode, split=options.split, transform=transform, train=True, download=True)
        te_dataset = datasets.EmnistDataset(DATAROOT, num_pulse=num_pulse, crop_class=[1,2,3,4,5,...], crop=crop, sampling=sampling, mode=options.mode, split=options.split, transform=transform, train=False, download=True, ori_img=True)
    
    # Fashion MNIST 数据集
    elif dataset == 'FMNIST':
        tr_dataset = datasets.FmnistDataset(DATAROOT, num_pulse=num_pulse, crop_class=[2,4,5,...], crop=crop, sampling=sampling, mode=options.mode, bin_thres=options.bin_threshold, transform=transform, train=True, download=True)
        te_dataset = datasets.FmnistDataset(DATAROOT, num_pulse=num_pulse, crop_class=[2,4,5,...], crop=crop, sampling=sampling, mode=options.mode, ori_img=True, transform=transform, train=False, download=True)

    # MNIST 数据集
    elif dataset == 'MNIST':
        tr_dataset = datasets.MnistDataset(DATAROOT, num_pulse, crop, sampling, mode=options.mode, ori_img=False, transform=transform, train=True, download=True)
        te_dataset = datasets.MnistDataset(DATAROOT, num_pulse, crop, sampling, mode=options.mode, ori_img=True, transform=transform, train=False, download=True)

    # 数据加载器
    train_loader = DataLoader(tr_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(te_dataset, batch_size=te_batchsize)

    num_class = tr_dataset.num_class  # 类别数量
    num_data = len(tr_dataset)  # 训练数据数量
    num_te_data = len(te_dataset)  # 测试数据数量

    new_img_width = tr_dataset.get_new_width()  # 获取图像的新宽度

    # 加载设备数据
    device_output = utils.oect_data_proc_std(path=device_path, device_test_cnt=device_tested_number)
    device_output = device_output.to_numpy().astype(float)

    # 初始化模型
    model = ann_model.model(new_img_width, num_class, batchsize, 1, conds_up, conds_down, a_w2c=config['a_w2c'], bias_w2c=config['bias_w2c'], config=config)

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)  # 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 学习率调度器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 选择训练功能或保存特征功能
    if choose_func == 'train_feat':
        model_path = train_funcs.train_with_feature(num_data, num_te_data, num_class, options.epoch, batchsize, te_batchsize, train_loader, test_loader, model, optimizer, scheduler, criterion, dataset, save_dir_name)
        return model_path
    elif choose_func == 'save_feat':
        filename = f'data/05s_{dataset}_' + datetime.strftime(t, '%m%d%H%M')
        train_funcs.save_rc_feature(train_loader, test_loader, num_pulse, device_output, device_tested_number, filename)
        return filename

# 主函数，执行训练、特征提取和模型测试
if __name__ == "__main__":
    # 加载脉冲数据
    conds_up = np.load('ann/single_up_cycle_50pulse_test_0326.npy')
    conds_down = np.load('ann/single_down_cycle_50pulse_test_0326.npy')
    conds = utils.conds_combine(conds_up, conds_down)
    bias_w2c = conds_up.mean()
    a_w2c = conds_down.std()

    # 记录日志
    utils.write_log(save_dir_name, options)

    # 模型配置参数
    params = {'lr': 1.01, 'a_w2c': conds_up.std(), 'bias_w2c': conds_up.mean(), 'weight_limit': 0.98562}
    feat_files = []
    feat_files_dict = {}
    model_paths = {}

    '''保存特征并使用特征进行训练'''
    for dataset in ['FMNIST', 'EMNIST', 'MNIST']:
        # 保存特征
        filename = train(options, params, dataset, 'save_feat')
        feat_files.append(filename)
        feat_files_dict[dataset] = f'{filename}_te.pt'

        # 使用保存的特征进行训练
        model_path = train(options, params, dataset, 'train_feat', f'{filename}_tr.pt', f'{filename}_te.pt', load_pt=True)
        model_paths[dataset] = model_path

    # 尺寸测试
    size_dataset = datasets.FashionWithMnist(roots_dict=feat_files_dict, soft=True)
    SizeDataLoader = DataLoader(size_dataset, batch_size=1, shuffle=False)
    num_te_data = len(size_dataset)

    # 加载和量化模型
    e_size_model, f_model, m_model = torch.load(model_paths['EMNIST']), torch.load(model_paths['FMNIST']), torch.load(model_paths['MNIST'])
    e_size_model = utils.model_quantize(e_size_model, conds, 0.999135, plot_weight=True, save_dir_name=save_dir_name)
    f_model = utils.model_quantize(f_model, conds, 0.99300577, plot_weight=True, save_dir_name=save_dir_name)
    m_model = utils.model_quantize(m_model, conds, 0.98562, plot_weight=True, save_dir_name=save_dir_name)

    # 测试模型尺寸
    train_funcs.test_fashion_size(num_te_data, te_batchsize, options.epoch, SizeDataLoader, f_model, m_model, e_size_model, device_tested_number, num_pulse, save_dir_name)
