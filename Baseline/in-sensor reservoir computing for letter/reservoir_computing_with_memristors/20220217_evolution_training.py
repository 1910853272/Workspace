# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter

# 定义筛选条件的lambda函数，用于选择文件名中包含'_'的.npy文件
filt = lambda x: '_' in x
# 定义提取字母名称的lambda函数，用于从文件名中获取字母的标识
get_letter = lambda x: x.split('.')[0]

# 获取所有.npy文件的列表，并筛选出文件名包含'_'的文件，表示这些文件与字母相关
letters = list(filter(filt, glob('*.npy')))

# 手动定义一个包含文件名的列表，用于指定10个文件来进行字母识别
letters = ['l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy', 'l5_p.npy',
           'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy', 'letsbuy.npy', 'letsgo.npy',
           'letsride.npy', 'kick.npy', 'getout.npy']

# 仅选取前10个文件用于训练模型
letters = letters[:10]
# 使用map函数将所有选定文件的数据加载成numpy数组
ls = list(map(np.load, letters))
# 将文件名（去掉后缀）作为键，对应数据作为值，创建一个字典存储字母的图像数据
d = dict(zip(list(map(get_letter, letters)), ls))

# %% 绘制用于识别的字母图像
# 再次定义字母列表，以便之后统一操作
letters = ['l0_ya.npy', 'l1_yu.npy', 'l2_oi.npy', 'l3_yoi.npy', 'l4_yai.npy',
           'l5_p.npy', 'l6_m.npy', 'l7_t.npy', 'l8_r.npy', 'l9_b.npy']

# 创建一个子图窗口，使用5行2列的布局展示字母图像
fig, ax = plt.subplots(len(letters) // 2, 2)
# 将子图展平成一维列表，方便逐个访问
ax = [a for ae in ax for a in ae]

# 循环加载每个字母的numpy文件并显示为图像
for i, lett in enumerate(letters):
    ax[i].imshow(np.load(lett), cmap=plt.cm.Greens, clim=[-1, 2])
    ax[i].axis('off')  # 关闭坐标轴显示

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()  # 显示字母图像


# %% 模拟memristor的行为
def output_row(initial_state, input_signal):
    """
    模拟memristor的行为，生成电导值的序列。

    参数
    ----------
    initial_state : float
        初始电导值。
    input_signal : array-like
        施加的脉冲信号序列。

    返回
    -------
    a : list
        电导输出的序列。
    """
    # 将初始电导值添加到输出序列中
    a = [initial_state]

    # 遍历输入信号，根据信号的状态调整电导值
    for i in range(5):
        if input_signal[i] > 0:
            # 如果输入信号大于0，应用exp增长，限制在0.1到1之间
            a.append(np.clip(a[i], 0.1, 1) * np.exp(1))
        else:
            # 如果输入信号小于等于0，应用其他变化，限制在1到10之间
            a.append(np.clip(a[i], 1, 10) * (3 - np.exp(1)))
    return np.array(a).flatten()  # 保证输出为一维数组


# 创建一个空矩阵用于存储输出数据
matrix = np.zeros((10, 31))

# 循环访问字典中的每个字母数据
for nl, lett in enumerate(d.keys()):
    for nr, row in enumerate(d[lett]):
        initial_state = np.random.random(1)  # 初始化一个随机的电导值
        output = output_row(initial_state, row)  # 计算输出值
        # 将输出值绘制到图表中
        ax[0].plot(output + np.random.random(1) * 1e-4, '-o')
        # 将输出值存储到矩阵中
        matrix[nl, nr * 6:(nr + 1) * 6] = output

# 在矩阵的最后一列添加一些随机数值
matrix[:, 30] = 2.5 * np.random.random((10,))

# 使用图像显示矩阵内容
fig, ax = plt.subplots()
cax = ax.imshow(matrix, extent=[10, 1, 1, 31], aspect='auto', cmap='viridis')
fig.colorbar(cax)
plt.show()  # 显示矩阵内容


# %% One-hot编码和softmax函数
def one_hot(y, c):
    """
    实现One-hot编码，将标签数据转换为one-hot格式

    参数
    ----------
    y : array-like
        标签数据。
    c : int
        类别数。

    返回
    -------
    y_hot : array
        转换后的one-hot编码矩阵。
    """
    # 创建一个 (m, c)大小的零矩阵
    y_hot = np.zeros((len(y), c))
    # 使用多维索引，在相应类别的位置设置为1
    y_hot[np.arange(len(y)), y] = 1
    return y_hot


def softmax(z):
    """
    实现softmax激活函数。

    参数
    ----------
    z : array-like
        输入的线性结果。

    返回
    -------
    exp : array
        经过softmax处理的概率分布。
    """
    # 为数值稳定性减去z的最大值
    exp = np.exp(z - np.max(z))
    # 对每个示例应用softmax
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp


# %% 定义模型训练函数
def fit(X, y, lr, c, epochs):
    """
    训练模型参数。

    参数
    ----------
    X : array
        输入数据。
    y : array
        真实标签。
    lr : float
        学习率。
    c : int
        类别数。
    epochs : int
        训练轮数。

    返回
    -------
    w : array
        学习到的权重矩阵。
    b : array
        学习到的偏置向量。
    losses : list
        每轮训练的损失值。
    """
    m, n = X.shape  # 获取样本数量和特征数量
    w = np.random.random((n, c))  # 随机初始化权重
    b = np.random.random(c)  # 随机初始化偏置
    losses = []  # 存储损失值的列表

    # np.save('initial_w.npy', w)  # 保存初始权重

    # 训练循环
    for epoch in range(epochs):
        z = X @ w + b  # 计算预测值
        y_hat = softmax(z)  # 应用softmax函数
        y_hot = one_hot(y, c)  # 转换标签为one-hot格式

        # 计算损失的梯度
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)

        # 更新权重和偏置
        w -= lr * w_grad
        b -= lr * b_grad

        # np.save('w' + str(epoch) + '.npy', w)  # 保存权重
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))  # 计算损失
        losses.append(loss)

        # 每轮输出损失值
        print(f'Epoch {epoch} ==> Loss = {loss}')
    return w, b, losses


# 训练模型并绘制损失曲线
X = np.zeros((10, 30))  # 创建输入矩阵
for i, letter in enumerate(d.keys()):
    initial_state = np.random.random(1)  # 初始化电导值
    output = []
    for row in d[letter]:
        output.append(output_row(initial_state, row))  # 生成输出行
    X[i, :] = np.concatenate(output)  # 将output中的数据合并成单一的一维数组

w, b, losses = fit(X, np.arange(10), 0.1, 10, 100)  # 训练模型
plt.figure()
plt.plot(losses)  # 绘制损失曲线
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()  # 显示损失曲线


# %% 预测函数
def predict(X, w, b):
    """
    使用训练好的模型进行预测。

    参数
    ----------
    X : array
        输入数据。
    w : array
        权重矩阵。
    b : array
        偏置向量。

    返回
    -------
    array
        预测的类别。
    """
    z = X @ w + b  # 计算线性结果
    y_hat = softmax(z)  # 应用softmax函数
    return np.argmax(y_hat, axis=1)  # 返回最大值的索引作为预测类别


# %% 混淆矩阵的生成与显示
X = np.zeros((25, 30))  # 创建输入矩阵
confusion_matrix = np.zeros((10, 10))  # 创建空混淆矩阵

# 循环每个字母，生成预测并更新混淆矩阵
for num_letter in range(10):
    for i in range(5):
        for j in range(5):
            test_letter = d[letters[num_letter].split('.')[0]].copy()  # 复制字母数据
            test_letter[i, j] = 1 if not test_letter[i, j] else 0  # 随机扰动数据
            initial_state = np.random.random(1)  # 初始电导
            output = []
            for row in np.array(test_letter):
                output.append(output_row(initial_state, row))  # 生成输出行
            X[5 * i + j, :] = np.concatenate(output)  # 将output中的数据合并成单一的一维数组

    predictions = predict(X, w, b)  # 预测当前字母
    for n_lett, prob in Counter(predictions).items():
        confusion_matrix[num_letter, n_lett] = prob / 25 * 100  # 更新混淆矩阵

plt.figure()
plt.imshow(confusion_matrix, cmap="viridis")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()  # 显示混淆矩阵
