import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import utility.utils as utils
from utility.utils import oect_data_proc_std
from utility.dvs_dataset import DvsTFDataset
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

'''OPTION'''
# 定义命令行参数，用于配置模型训练的相关参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-9)
parser.add_argument('--device_name', type=str, default='p_NDI_05s')
parser.add_argument('--device_cnt', type=int, default=1)
parser.add_argument('--feat_path', type=str, default='11222014_final_012cls') # 4 class version
parser.add_argument('--log_dir', type=str, default='')
options = parser.parse_args()

num_cls = 3  # 定义类别数量
img_width = 28  # 图像宽度（28x28）

'''PATH'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的根目录
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')  # 数据集目录
TR_FILEPATH = os.path.join(DATA_DIR, 'dvs_proced_tr_012cls_w28_32frame_s6i5.pt')  # 训练集路径
TE_FILEPATH = os.path.join(DATA_DIR, 'dvs_proced_te_012cls_w28_32frame_s6i5.pt')  # 测试集路径
DEVICE_DIR = os.path.join(ROOT_DIR, 'data')  # 设备数据目录

SAVE_PATH = os.path.join(ROOT_DIR, 'log/dvs')  # 日志保存路径
time_str = datetime.now().strftime('%m%d%H%M')  # 获取当前时间戳
savepath = os.path.join(SAVE_PATH, f'{options.log_dir}{time_str}')  # 最终保存路径

# 检查保存路径是否存在，不存在则创建
for path in [SAVE_PATH, savepath]:
    if not os.path.exists(path):
        os.mkdir(path)

'''load dataset'''
# 加载训练集和测试集数据
tr_dataset = DvsTFDataset(TR_FILEPATH)
te_dataset = DvsTFDataset(TE_FILEPATH)

# 获取训练集样本数量，并创建 DataLoader
num_tr_data = len(tr_dataset)
tr_loader = DataLoader(tr_dataset, batch_size=num_tr_data, shuffle=False, num_workers=0)
te_loader = DataLoader(te_dataset, batch_size=options.batch, shuffle=False)

'''load device data'''
# 加载设备数据
device_path = os.path.join(DEVICE_DIR, f'{options.device_name}.xlsx')
device_output = oect_data_proc_std(path=device_path, device_test_cnt=options.device_cnt)
device_output = device_output.to_numpy().astype(np.float32)

'''define model'''
# 定义简单的线性模型
model = nn.Sequential(nn.Linear(in_features=img_width ** 2, out_features=num_cls))
optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
cross_entropy = nn.CrossEntropyLoss()

def feat_extract(savepath=''):
    '''特征提取函数'''
    print('Extracting feature')
    tr_feat = []
    for i, (data, label) in enumerate(tr_loader):
        this_batch_size = data.shape[0]
        data = data.view(this_batch_size, -1, img_width ** 2)
        oect_output = utils.batch_rc_feat_extract_in_construction(data, device_output, options.device_cnt,
                                                    1, 5, this_batch_size)
        tr_feat.append(oect_output)
        del oect_output

    te_feat = []
    for i, (data, label) in enumerate(te_loader):
        this_batch_size = data.shape[0]
        data = data.view(this_batch_size, -1, img_width ** 2)
        oect_output = utils.batch_rc_feat_extract(data, device_output, options.device_cnt,
                                                    5, this_batch_size)
        te_feat.append(oect_output)

    if savepath:
        torch.save((tr_feat, te_feat), os.path.join(savepath, 'feat.pt'))
    return tr_feat, te_feat

# 尝试加载已提取的特征，如果不存在则提取
try:
    tr_feat, te_feat = torch.load(f'log/dvs/{options.feat_path}/feat.pt')
    print('Use extracted feature')
except:
    print('No existing feature, extract feature from dataset')
    tr_feat, te_feat = feat_extract(savepath=savepath)

'''training'''
print('start training')
for epoch in range(1):
    correct_cnt, loss_epoch = 0, 0
    for i, (data, label) in enumerate(tr_loader):
        this_batch_size = data.shape[0]
        oect_output = torch.cat(tr_feat, dim=0)
        label_onehot = torch.zeros(label.shape[0], num_cls).scatter_(1, label.view(-1, 1).to(torch.long), 1)
        readout = torch.linalg.lstsq(oect_output, label_onehot.to(oect_output.dtype)).solution

    correct_cnt = 0
    for i, (data, label) in enumerate(te_loader):
        this_batch_size = data.shape[0]
        oect_output = te_feat[i]
        logic = torch.mm(oect_output, readout)
        logic = torch.round(logic)
        correct_cnt += torch.sum(torch.argmax(logic, dim=1) == label)
    te_acc = correct_cnt / len(te_dataset)
    print(f'epoch: {epoch}, tr loss: {loss_epoch}, te acc: {te_acc}')

# PCA 降维并绘制二维散点图
color = ['coral', 'dodgerblue', 'tan', 'orange', 'green', 'silver', 'chocolate', 'lightblue', 'violet', 'crimson']
color_list = [color[i] for i in labels]
pca = PCA(n_components=2)
outputs_pca = pca.fit_transform(torch.cat(te_feat, 0))
plt.scatter(outputs_pca[:, 0], outputs_pca[:, 1], c=color_list)
plt.savefig(f'{savepath}/dvs_2cls_pca.pdf')
plt.close()

# 生成混淆矩阵并绘制
conf_mat = confusion_matrix(labels, outputs)
confusion_matrix_df = pd.DataFrame(conf_mat, index=range(num_cls), columns=range(num_cls))
plt.figure(figsize=(num_cls, num_cls))
sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.savefig(f'{savepath}/conf_mat_dvs.pdf', format='pdf')
plt.close()

normed_conf_mat = conf_mat / np.expand_dims(conf_mat.sum(1), -1)
normed_confusion_matrix_df = pd.DataFrame(normed_conf_mat, index=range(num_cls), columns=range(num_cls))
plt.figure()
sns.heatmap(normed_confusion_matrix_df, annot=True, fmt='.2f', cmap=plt.cm.Blues)
plt.savefig(f'{savepath}/normed_conf_mat_dvs.pdf', format='pdf')
plt.close()
print('Confusion matrix saved')

if __name__ == '__main__':
    pass