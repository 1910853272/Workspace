import matplotlib.pyplot as plt  # 绘图库，用于绘制训练过程曲线和混淆矩阵

import os  # 用于文件路径和目录操作
import pandas as pd  # 用于数据框处理，以及读取/保存 CSV 文件（未在本代码中直接使用，可根据需要扩展）
import numpy as np  # 用于科学计算，如数组操作、数值计算
import random  # 用于随机数生成，用于从 PSI 数据中随机选择值
import shutil  # 用于文件和目录的管理，如复制、移动文件（可选扩展）
import imageio  # 用于生成 GIF 动画（可选扩展）
from PIL import Image  # 用于图像读写与处理（可选扩展）
from IPython.display import clear_output  # 在 Jupyter Notebook 中用于清除单元格输出
from tqdm import tqdm  # 进度条显示，方便跟踪循环执行进度

# 以下库用于构建和训练前馈神经网络模型
from sklearn.preprocessing import StandardScaler  # 标准化特征
from sklearn.preprocessing import OneHotEncoder  # 将类别标签进行独热编码
from keras.models import Sequential  # 顺序模型，用于按层构建神经网络
from keras.layers import Dense  # 全连接层，用于定义网络层
from sklearn.metrics import confusion_matrix  # 混淆矩阵计算函数
from sklearn.manifold import TSNE  # t-SNE 降维可视化（未在本代码中直接使用，可根据需要扩展）


def import_mnist_offline(file_path):
    """
    离线导入 MNIST 数据集
    参数:
        file_path: .npz 文件所在路径，包含 x_train, y_train, x_test, y_test 四个数组
    返回:
        (x_train, y_train), (x_test, y_test)
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Make sure the file is in the correct location or provide the full path.")
        # 如果文件不存在，则返回 None，便于上层判断
        return None, None
    else:
        # 使用 numpy.load 载入 .npz 压缩文件
        with np.load(file_path, allow_pickle=True) as data:
            x_train = data['x_train']  # 训练集输入
            y_train = data['y_train']  # 训练集标签
            x_test = data['x_test']    # 测试集输入
            y_test = data['y_test']    # 测试集标签
    return (x_train, y_train), (x_test, y_test)


def value_at_time_x_numpy(single_data, time):
    """
    在给定的时间向量中查找最接近目标时间的值
    参数:
        single_data: 二维数组，第一列为时间，第二列为对应值
        time: 目标时间点
    返回:
        (value, closest_time, index)
    """
    # 将时间和对应值转换为 Python 列表，便于后续操作
    time_vector = single_data[:, 0].tolist()
    value_vector = single_data[:, 1].tolist()
    # 找到与目标 time 最接近的时间值
    closest_x_value = min(time_vector, key=lambda x: abs(time - x))
    # 获取该时间值在列表中的索引
    idx_closest_x_value = time_vector.index(closest_x_value)
    # 返回对应的值、最接近时间点、本索引
    return value_vector[idx_closest_x_value], closest_x_value, idx_closest_x_value


def find_nearest_value_index(array, target_value):
    """
    在一维 numpy 数组中查找与目标值最接近的索引
    参数:
        array: numpy 一维数组
        target_value: 目标数值
    返回:
        index: 最近似值的索引
    """
    index = (np.abs(array - target_value)).argmin()
    return index


def assign_experimental_data_with_digit(
        data_at_specific_time,
        digit_rows,
        num_digits_train,
        num_digits_test,
        digit_train,
        digit_test,
        selector_reservoir=0):
    """
    根据数字二进制数据，为 PSI (physical system input) 分配实验数据
    参数:
        data_at_specific_time: 字典，键为二进制字符串，值为对应时间上的 PSI 数据列表
        digit_rows: 每个数字包含的二进制行数
        num_digits_train: 训练集数字样本数
        num_digits_test: 测试集数字样本数
        digit_train: 训练集的二进制数组列表
        digit_test: 测试集的二进制数组列表
        selector_reservoir: 选择策略，0~4，不同策略对应不同的数据映射方式
    返回:
        digit_train_reduced_np: 训练集实验数据矩阵, shape=(num_digits_train, digit_rows)
        digit_test_reduced_np: 测试集实验数据矩阵, shape=(num_digits_test, digit_rows)
    """
    digit_cols_reduced = 1  # 简化后的列数为 1
    # 初始化存储容器，列表中每个元素是 (digit_rows x digit_cols_reduced) 的零矩阵
    digit_train_reduced = [np.zeros((digit_rows, digit_cols_reduced)) for _ in range(num_digits_train)]
    digit_test_reduced = [np.zeros((digit_rows, digit_cols_reduced)) for _ in range(num_digits_test)]

    # 遍历训练集每个数字样本
    for idx_digit, single_digit in tqdm(enumerate(digit_train), desc='Training set', total=num_digits_train):
        for idx_row, single_row in enumerate(single_digit):
            # 将单行二进制数组转换为字符串
            eq_binary_number = ''.join(str(int(x)) for x in single_row)
            # 根据 selector_reservoir 选择不同的映射方式
            if selector_reservoir == 0:
                # 随机选择同一时间点上的 PSI 数据
                digit_train_reduced[idx_digit][idx_row] = random.choice(data_at_specific_time[eq_binary_number])
            elif selector_reservoir == 1:
                # 取最后一个二进制值作为 PSI 输入
                digit_train_reduced[idx_digit][idx_row] = single_row[-1]
            elif selector_reservoir == 2:
                # 将二进制字符串转为十进制数
                digit_train_reduced[idx_digit][idx_row] = int(eq_binary_number, 2)
            elif selector_reservoir == 3:
                # 取该时间点 PSI 数据的均值
                digit_train_reduced[idx_digit][idx_row] = np.mean(data_at_specific_time[eq_binary_number])
            else:
                # 其他情况填充 None
                digit_train_reduced[idx_digit][idx_row] = None

    # 同样地，遍历测试集样本
    for idx_digit, single_digit in tqdm(enumerate(digit_test), desc='Testing set', total=num_digits_test):
        for idx_row, single_row in enumerate(single_digit):
            eq_binary_number = ''.join(str(int(x)) for x in single_row)
            if selector_reservoir == 0:
                digit_test_reduced[idx_digit][idx_row] = random.choice(data_at_specific_time[eq_binary_number])
            elif selector_reservoir == 1:
                digit_test_reduced[idx_digit][idx_row] = single_row[-1]
            elif selector_reservoir == 2:
                digit_test_reduced[idx_digit][idx_row] = int(eq_binary_number, 2)
            elif selector_reservoir == 3:
                digit_test_reduced[idx_digit][idx_row] = np.mean(data_at_specific_time[eq_binary_number])
            else:
                digit_test_reduced[idx_digit][idx_row] = None

    # 将列表转换为 numpy 矩阵，方便后续训练输入
    digit_train_reduced_np = np.array(digit_train_reduced).reshape(num_digits_train, digit_rows)
    digit_test_reduced_np = np.array(digit_test_reduced).reshape(num_digits_test, digit_rows)
    return digit_train_reduced_np, digit_test_reduced_np


def perform_training(
        start_train,
        start_test,
        num_digits_train,
        num_digits_test,
        digit_train_class,
        digit_test_class,
        epochs,
        batch_size,
        digit_train_reduced_np,
        digit_test_reduced_np,
        verbose_training):
    """
    构建并训练一个简单的前馈神经网络，并返回训练历史及评估指标
    参数:
        start_train, start_test: 训练/测试切片起始索引
        num_digits_train/test: 样本数量
        digit_train_class/test_class: 样本标签
        epochs, batch_size: 训练超参数
        digit_train_reduced_np, digit_test_reduced_np: 实验输入矩阵
        verbose_training: 是否显示训练进度
    返回:
        history: 训练过程历史记录
        conf_matrix: 原始混淆矩阵
        conf_matrix_norm: 归一化混淆矩阵
        test_accuracy: 测试集准确率
        y_pred_class: 预测类别列表
        test_out: 真实类别列表
    """
    # 提取输入输出
    train_in = digit_train_reduced_np[start_train:start_train+num_digits_train, :]
    test_in = digit_test_reduced_np[start_test:start_test+num_digits_test, :]
    train_out = digit_train_class[start_train:start_train+num_digits_train].reshape(num_digits_train, 1)
    test_out = digit_test_class[start_test:start_test+num_digits_test].reshape(num_digits_test, 1)

    # 特征标准化和标签独热编码
    scaler = StandardScaler().fit(train_in)
    train_in = scaler.transform(train_in)
    encoder = OneHotEncoder().fit(train_out)
    train_out = encoder.transform(train_out).toarray()

    # 构建顺序模型，仅一层全连接 softmax，用于 10 类分类
    model = Sequential()
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 开始训练
    history = model.fit(train_in, train_out, epochs=epochs, batch_size=batch_size, verbose=verbose_training)

    # 对测试集进行标准化
    test_in = scaler.transform(test_in)
    # 获取预测结果
    y_pred = model.predict(test_in)
    pred = [np.argmax(y) for y in y_pred]
    # 计算测试准确率
    test_accuracy = np.mean([1 if p == int(true) else 0 for p, true in zip(pred, test_out)])

    # 计算混淆矩阵，归一化和未归一化
    conf_matrix = confusion_matrix(test_out, pred, normalize=None)
    conf_matrix_norm = confusion_matrix(test_out, pred, normalize='true')

    y_pred_class = pred
    return history, conf_matrix, conf_matrix_norm, test_accuracy, y_pred_class, test_out


def plot_training_data(out_dir_training_outputs, history, conf_matrix, conf_matrix_norm, repetition):
    """
    绘制并保存训练准确率、损失曲线，以及混淆矩阵热图
    参数:
        out_dir_training_outputs: 输出目录
        history: 训练历史记录
        conf_matrix: 未归一化混淆矩阵
        conf_matrix_norm: 归一化混淆矩阵
        repetition: 实验重复标识，用于文件命名
    """
    # 绘制训练准确率曲线
    plt.figure()
    plt.plot(history.history['accuracy'], linewidth=2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(out_dir_training_outputs, f'model_accuracy_rep{repetition}.png'), dpi=1200)

    # 保存准确率数据到文本
    np.savetxt(os.path.join(out_dir_training_outputs, f'model_accuracy_rep{repetition}.txt'), history.history['accuracy'])

    # 绘制训练损失曲线
    plt.figure()
    plt.plot(history.history['loss'], linewidth=2)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(out_dir_training_outputs, f'model_loss_rep{repetition}.png'), dpi=1200)
    np.savetxt(os.path.join(out_dir_training_outputs, f'model_loss_rep{repetition}.txt'), history.history['loss'])

    # 绘制混淆矩阵热图
    plt.figure(figsize=(18, 9))
    plt.title('Confusion Matrix', fontsize=25)
    plt.imshow(conf_matrix_norm, cmap='Blues')
    plt.colorbar(label='%')
    plt.xlabel('PREDICTED', fontsize=20)
    plt.ylabel('TRUE', fontsize=20)
    ticks = range(conf_matrix.shape[0])
    plt.xticks(ticks, ticks, fontsize=15)
    plt.yticks(ticks, ticks, fontsize=15, rotation=90)
    # 在热图上添加数值标签
    for i in ticks:
        for j in ticks:
            if conf_matrix[i, j] != 0:
                plt.text(j, i, str(conf_matrix[i, j]),
                         ha='center', va='center', color='white', fontsize=15)
    plt.savefig(os.path.join(out_dir_training_outputs, f'conf_matrix_rep{repetition}.png'), dpi=1200)
    plt.show()
