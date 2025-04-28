import time
import os
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import utils


def get_dataset_statics(num_data,
                        batchsize,
                        data_loader):
    '''
    计算数据集的统计信息：均值、标准差、最大值和最小值
    '''
    mean_list = []  # 存储每个batch的均值
    std_list = []  # 存储每个batch的标准差
    data_max, data_min = 0, 0  # 初始化最大值和最小值
    for data, target in tqdm(data_loader):  # 遍历数据加载器中的数据
        data = data.to(torch.float)  # 将数据转换为浮点型
        mean = data.to(torch.float).mean(dim=-1).sum(0)  # 计算每个样本的均值
        std = data.to(torch.float).std(dim=-1).sum(0)  # 计算每个样本的标准差
        max = data.max()  # 计算当前batch的最大值
        min = data.min()  # 计算当前batch的最小值
        mean_list.append(mean)  # 将均值加入列表
        std_list.append(std)  # 将标准差加入列表
        data_max = max if max > data_max else data_max  # 更新全局最大值
        data_min = min if min < data_min else data_min  # 更新全局最小值
    mean = sum(mean_list) / num_data  # 计算数据集的均值
    std = sum(std_list) / num_data  # 计算数据集的标准差

    # 输出数据集的统计信息
    print(f'mean: {mean}, std: {std}, max: {data_max}, min: {data_min}')


def train_end_to_end(device_output,
                     num_data,
                     num_epoch,
                     batchsize,
                     train_loader,
                     model,
                     optimizer,
                     scheduler,
                     criterion,
                     device_tested_number,
                     num_pulse,
                     save_dir_name):
    '''
    端到端训练模型
    '''
    start_time = time.time()  # 记录训练开始时间

    # 初始化训练过程中的记录列表
    acc_list = []  # 存储每个epoch的准确率
    loss_list = []  # 存储每个epoch的损失值
    log_list = []  # 存储日志信息

    # 迭代进行训练
    for epoch in range(num_epoch):
        acc = []  # 存储当前epoch的每个batch的准确率
        loss = 0  # 初始化损失
        for i, (data, target) in enumerate(train_loader):  # 遍历训练数据
            optimizer.zero_grad()  # 清零梯度

            this_batch_size = len(data)  # 获取当前batch的大小

            # 通过OECT提取特征
            oect_output = utils.batch_rc_feat_extract(data,
                                                    device_output,
                                                    device_tested_number,
                                                    num_pulse,
                                                    this_batch_size)

            # 通过模型进行预测
            logic = F.softmax(model(oect_output), dim=-1)

            # 计算损失
            batch_loss = criterion(logic, target)
            loss += batch_loss

            # 计算当前batch的准确率
            batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
            acc.append(batch_acc)

            # 反向传播
            batch_loss.backward()
            optimizer.step()  # 更新参数

        scheduler.step()  # 调整学习率
        acc_epoch = (sum(acc) * batchsize / num_data).numpy()  # 计算当前epoch的准确率
        acc_list.append(acc_epoch)
        loss_list.append(loss)

        epoch_end_time = time.time()  # 记录当前epoch结束的时间
        if epoch == 0:
            epoch_time = epoch_end_time - start_time  # 第一个epoch的时间
        else:
            epoch_time = epoch_end_time - epoch_start_time  # 后续epoch的时间
        epoch_start_time = epoch_end_time  # 更新epoch的开始时间

        # 打印当前epoch的训练信息
        log = "epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time)
        print(log)
        log_list.append(log + '\n')  # 保存日志

    utils.write_log(save_dir_name, log_list)  # 将训练日志写入文件

    # 保存训练后的模型
    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))


def train_with_feature(num_data,
                       num_epoch,
                       batchsize,
                       train_loader,
                       model,
                       optimizer,
                       scheduler,
                       criterion,
                       save_dir_name):
    '''
    使用特征训练模型（提取后的特征输入）
    '''
    start_time = time.time()  # 记录训练开始时间
    acc_list = []  # 存储每个epoch的准确率
    loss_list = []  # 存储每个epoch的损失值
    log_list = []  # 存储日志信息

    # 迭代进行训练
    for epoch in range(num_epoch):
        acc = []  # 存储当前epoch的每个batch的准确率
        loss = 0  # 初始化损失
        for i, (data, target) in enumerate(train_loader):  # 遍历训练数据
            optimizer.zero_grad()  # 清零梯度

            this_batch_size = len(data)  # 获取当前batch的大小

            data = data.to(torch.float).squeeze()  # 将数据转换为浮点型并去除多余的维度

            # 通过模型进行预测
            logic = F.softmax(model(data), dim=-1)

            # 计算损失
            batch_loss = criterion(logic, target)
            loss += batch_loss

            # 计算当前batch的准确率
            batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
            acc.append(batch_acc)

            # 反向传播
            batch_loss.backward()
            optimizer.step()  # 更新参数

        scheduler.step()  # 调整学习率
        acc_epoch = (sum(acc) * batchsize / num_data).numpy()  # 计算当前epoch的准确率
        acc_list.append(acc_epoch)
        loss_list.append(loss)

        epoch_end_time = time.time()  # 记录当前epoch结束的时间
        if epoch == 0:
            epoch_time = epoch_end_time - start_time  # 第一个epoch的时间
        else:
            epoch_time = epoch_end_time - epoch_start_time  # 后续epoch的时间
        epoch_start_time = epoch_end_time  # 更新epoch的开始时间

        # 打印当前epoch的训练信息
        log = "epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time)
        print(log)
        log_list.append(log + '\n')  # 保存日志

    utils.write_log(save_dir_name, log_list)  # 将训练日志写入文件

    # 保存训练后的模型
    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))


def test(tran_type,
         device_output,
         device_tested_number,
         num_data,
         num_pulse,
         num_class,
         batchsize,
         test_loader,
         model,
         save_dir_name,
         img_save=False):
    '''
    测试模型性能，并生成混淆矩阵
    '''
    te_accs = []  # 存储测试集上的准确率
    te_outputs = []  # 存储测试集上的预测输出
    targets = []  # 存储真实标签
    with torch.no_grad():  # 不计算梯度
        for i, (data, img, target) in enumerate(test_loader):  # 遍历测试数据
            this_batch_size = len(data)

            if tran_type == 'train_ete':  # 如果是训练端到端类型
                data = utils.batch_rc_feat_extract(data,
                                                  device_output,
                                                  device_tested_number,
                                                  num_pulse,
                                                  this_batch_size)
            data = data.to(torch.float)  # 将数据转换为浮点型
            output = F.softmax(model(data.squeeze()), dim=-1)  # 进行预测
            te_outputs.append(output)  # 存储预测输出
            acc = torch.sum(output.argmax(dim=-1) == target) / this_batch_size  # 计算准确率
            te_accs.append(acc)
            targets.append(target)  # 存储真实标签

        te_acc = (sum(te_accs) * batchsize / num_data).numpy()  # 计算整体准确率

        # 输出测试结果
        log = "test acc: %.6f" % te_acc + '\n'
        print(log)
        utils.write_log(save_dir_name, log)

        # 将测试输出转换为tensor
        te_outputs = torch.stack(te_outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # 生成混淆矩阵
        conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=-1))

        # 将混淆矩阵转为DataFrame格式
        conf_mat_dataframe = pd.DataFrame(conf_mat,
                                        index=list(range(num_class)),
                                        columns=list(range(num_class)))

        # 计算归一化混淆矩阵
        conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

        # 绘制混淆矩阵并保存
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat_dataframe, annot=True, fmt='d')
        plt.savefig(os.path.join(save_dir_name, f'conf_mat_{te_acc * 1e4:.0f}'))
        plt.close()

        # 绘制归一化混淆矩阵并保存
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat_normalized, annot=True)
        plt.savefig(os.path.join(save_dir_name, f'conf_mat_normalized_{te_acc * 1e4:.0f}'))
        plt.close()
        print('confusion matrix saved')


def save_rc_feature(train_loader,
                    test_loader,
                    num_pulse,
                    device_output,
                    device_tested_number,
                    filename):
    '''
    提取并保存训练和测试集的RC特征
    '''
    device_features = []  # 存储训练集的特征
    tr_targets = []  # 存储训练集的目标标签

    for i, (data, target) in enumerate(train_loader):  # 遍历训练集
        data = data.squeeze()
        oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
        device_features.append(oect_output)
        tr_targets.append(target)
    tr_features = torch.stack(device_features, dim=0)
    tr_targets = torch.stack(tr_targets).squeeze()

    # 保存训练集特征和标签
    tr_filename = filename + f'_tr.pt'
    torch.save((tr_features, tr_targets), tr_filename)

    te_oect_outputs = []  # 存储测试集的特征
    te_targets = []  # 存储测试集的目标标签
    for i, (data, im, target) in enumerate(test_loader):  # 遍历测试集
        data = data.squeeze()
        oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
        te_oect_outputs.append(oect_output)
        te_targets.append(target)
    te_features = torch.stack(te_oect_outputs, dim=0)
    te_targets = torch.stack(te_targets).squeeze()

    # 保存测试集特征和标签
    te_filename = filename + f'_te.pt'
    torch.save((te_features, te_targets), te_filename)
