<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
DBPPD_ANN 属于 DBPD 模拟
创建于 4 月 8 日

@author: Wang Jialiang
"""

"""
导入包
"""
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from __Mod_route__ import get_root

# 获取根目录路径
rr = get_root()
Task_name = '\\20230927'
Task_subname = '_mixed_dataset_recog'

"""
定义函数
"""


# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义 Softmax 函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 定义标准化函数（未完成）
def norm(x):
    for i in x.shape[2]:
        for j in x.shape[1]:
            pass
    return


"""
加载输入数据
"""
# %%
# 设置模型路径
Model_r = rr + '\\Models' + Task_name + Task_subname
# 加载训练集和测试集的数据
Input_neurons_OG = np.load(Model_r + '\\Train_set_DBPD_Input_neurons.npy')
Labels_OG = np.load(Model_r + '\\Train_set_Labels.npy')
Input_neurons_test = np.load(Model_r + '\\Test_set_DBPD_Input_neurons.npy')
Labels_test = np.load(Model_r + '\\Test_set_Labels.npy')

# 转换标签为整数类型
Labels_OG = Labels_OG.astype(np.int64)
Labels_test = Labels_test.astype(np.int64)
# %%


"""
神经网络框架
"""
# %%
# 定义神经网络架构参数
input_size = 28 * 28
hidden_1_size = 200
hidden_2_size = 200
hidden_size = hidden_1_size + hidden_2_size
output_size = 10
learning_rate = 0.01

# 随机初始化隐藏层和输出层的权重和偏置
hidden_1_weights = np.random.randn(input_size, hidden_1_size) * np.sqrt(2 / input_size)
hidden_1_biases = np.zeros((1, hidden_1_size))
hidden_2_weights = np.random.randn(input_size, hidden_2_size) * np.sqrt(2 / input_size)
hidden_2_biases = np.zeros((1, hidden_2_size))
output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
output_biases = np.zeros((1, output_size))
# %%

"""
神经网络构建
"""
# %%
# 设置训练参数
num_epochs = 20
batch_size = 90
num_batches = len(Labels_OG) // batch_size
# 创建用于保存准确率和预测结果的数据框
accuracy = pd.DataFrame(index=range(num_epochs + 1),
                        columns=['Sce_1', 'Sce_2_day', 'Sce_2_night', 'Sce_2_interference'], dtype='float64')
predict_result = pd.DataFrame(index=range(5000),
                              columns=['Sce_1_labels', 'Sce_1_predicts',
                                       'Sce_2_day_labels', 'Sce_2_day_predicts',
                                       'Sce_2_night_labels', 'Sce_2_night_predicts',
                                       'Sce_2_interference_labels', 'Sce_2_interference_predicts'])

# 评估初始神经网络（随机参数）的准确率
# %%
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 0] = np.mean(predicted_labels == Labels_test[0:5000])
print('\n 初始参数在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[0, 0] * 100) + '%')

# 场景 2（白天）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 1] = np.mean(predicted_labels == Labels_test[5074:10074])
print('\n 初始参数在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[0, 1] * 100) + '%')

# 场景 2（夜晚）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 2] = np.mean(predicted_labels == Labels_test[10074:15074])
print('\n 初始参数在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[0, 2] * 100) + '%')

# 场景 2（干扰）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 3] = np.mean(predicted_labels == Labels_test[15074:20074])
print('\n 初始参数在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[0, 3] * 100) + '%')
# %%

# 训练神经网络
for epoch in range(num_epochs):

    print('\n 开始第 #' + str(epoch) + ' 轮训练')
    # 打乱训练集
    perm = np.random.permutation(len(Labels_OG))
    Labels_train = Labels_OG[perm]
    Input_neurons = Input_neurons_OG[perm]

    # 在小批量数据上训练
    print('\n 小批量数据训练中... \n')
    for i in tqdm(range(num_batches)):
        # 选择一个小批量
        start = i * batch_size
        end = start + batch_size
        batch_images = Input_neurons[start:end, :, :]
        batch_labels = Labels_train[start:end]

        # 前向传播
        hidden_1_activations = sigmoid(np.dot(batch_images[:, :, 0], hidden_1_weights) + hidden_1_biases)
        hidden_2_activations = sigmoid(np.dot(batch_images[:, :, 1], hidden_2_weights) + hidden_2_biases)
        hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
        output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)

        # 反向传播：计算损失函数的梯度
        output_grads = output_activations
        output_grads[np.arange(len(batch_labels)), batch_labels] -= 1
        output_grads /= len(batch_labels)
        # 反向传播：计算隐藏层的梯度
        hidden_grads = np.dot(output_grads, output_weights.T) * hidden_activations * (1 - hidden_activations)
        # 分为两个独立的部分
        hidden_1_grads = hidden_grads[:, 0:hidden_1_size]
        hidden_2_grads = hidden_grads[:, hidden_1_size:hidden_size]

        # 更新权重和偏置
        output_weights -= learning_rate * np.dot(hidden_activations.T, output_grads)
        output_biases -= learning_rate * np.sum(output_grads, axis=0, keepdims=True)
        hidden_1_weights -= learning_rate * np.dot(batch_images[:, :, 0].T, hidden_1_grads)
        hidden_2_weights -= learning_rate * np.dot(batch_images[:, :, 1].T, hidden_2_grads)
        hidden_1_biases -= learning_rate * np.sum(hidden_1_grads, axis=0, keepdims=True)
        hidden_2_biases -= learning_rate * np.sum(hidden_2_grads, axis=0, keepdims=True)

    # 评估模型在场景 1 测试集上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_1_labels'] = Labels_test[0:5000]
    predict_result['Sce_1_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 0] = np.mean(predict_result['Sce_1_labels'] == predict_result['Sce_1_predicts'])
    print('\n 模型在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[epoch + 1, 0] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（白天）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_day_labels'] = Labels_test[5074:10074]
    predict_result['Sce_2_day_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 1] = np.mean(predict_result['Sce_2_day_labels'] == predict_result['Sce_2_day_predicts'])
    print('\n 模型在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 1] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（夜晚）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_night_labels'] = Labels_test[10074:15074]
    predict_result['Sce_2_night_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 2] = np.mean(
        predict_result['Sce_2_night_labels'] == predict_result['Sce_2_night_predicts'])
    print('\n 模型在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 2] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（干扰）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_interference_labels'] = Labels_test[15074:20074]
    predict_result['Sce_2_interference_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 3] = np.mean(
        predict_result['Sce_2_interference_labels'] == predict_result['Sce_2_interference_predicts'])
    print('\n 模型在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 3] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')
# %%

"""
导出模型
"""
# %%
Model_series = 'DBPD_mixed_Gen_5'
outdir_models = Model_r + '\\' + Model_series
if not os.path.exists(outdir_models):
    os.mkdir(outdir_models)
# 保存模型权重和偏置
np.save(outdir_models + '\output_weights.npy', output_weights)
np.save(outdir_models + '\output_biases.npy', output_biases)
np.save(outdir_models + '\hidden_1_weights.npy', hidden_1_weights)
np.save(outdir_models + '\hidden_2_weights.npy', hidden_2_weights)
np.save(outdir_models + '\hidden_1_biases.npy', hidden_1_biases)
np.save(outdir_models + '\hidden_2_biases.npy', hidden_2_biases)
accuracy.to_csv(outdir_models + '\\Accuracy_evolution.csv')

# 保存预测结果
outdir_results = rr + '\\Experiments' + Task_name + Task_subname
if not os.path.exists(outdir_results):
    os.mkdir(outdir_results)
predict_result.to_csv(outdir_results + '\\Predict_results_' + Model_series + '.csv')
accuracy.to_csv(outdir_results + '\\Accuracy_evolution_' + Model_series + '.csv')
=======
# -*- coding: utf-8 -*-
"""
DBPPD_ANN 属于 DBPD 模拟
创建于 4 月 8 日

@author: Wang Jialiang
"""

"""
导入包
"""
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from __Mod_route__ import get_root

# 获取根目录路径
rr = get_root()
Task_name = '\\20230927'
Task_subname = '_mixed_dataset_recog'

"""
定义函数
"""


# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义 Softmax 函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 定义标准化函数（未完成）
def norm(x):
    for i in x.shape[2]:
        for j in x.shape[1]:
            pass
    return


"""
加载输入数据
"""
# %%
# 设置模型路径
Model_r = rr + '\\Models' + Task_name + Task_subname
# 加载训练集和测试集的数据
Input_neurons_OG = np.load(Model_r + '\\Train_set_DBPD_Input_neurons.npy')
Labels_OG = np.load(Model_r + '\\Train_set_Labels.npy')
Input_neurons_test = np.load(Model_r + '\\Test_set_DBPD_Input_neurons.npy')
Labels_test = np.load(Model_r + '\\Test_set_Labels.npy')

# 转换标签为整数类型
Labels_OG = Labels_OG.astype(np.int64)
Labels_test = Labels_test.astype(np.int64)
# %%


"""
神经网络框架
"""
# %%
# 定义神经网络架构参数
input_size = 28 * 28
hidden_1_size = 200
hidden_2_size = 200
hidden_size = hidden_1_size + hidden_2_size
output_size = 10
learning_rate = 0.01

# 随机初始化隐藏层和输出层的权重和偏置
hidden_1_weights = np.random.randn(input_size, hidden_1_size) * np.sqrt(2 / input_size)
hidden_1_biases = np.zeros((1, hidden_1_size))
hidden_2_weights = np.random.randn(input_size, hidden_2_size) * np.sqrt(2 / input_size)
hidden_2_biases = np.zeros((1, hidden_2_size))
output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
output_biases = np.zeros((1, output_size))
# %%

"""
神经网络构建
"""
# %%
# 设置训练参数
num_epochs = 20
batch_size = 90
num_batches = len(Labels_OG) // batch_size
# 创建用于保存准确率和预测结果的数据框
accuracy = pd.DataFrame(index=range(num_epochs + 1),
                        columns=['Sce_1', 'Sce_2_day', 'Sce_2_night', 'Sce_2_interference'], dtype='float64')
predict_result = pd.DataFrame(index=range(5000),
                              columns=['Sce_1_labels', 'Sce_1_predicts',
                                       'Sce_2_day_labels', 'Sce_2_day_predicts',
                                       'Sce_2_night_labels', 'Sce_2_night_predicts',
                                       'Sce_2_interference_labels', 'Sce_2_interference_predicts'])

# 评估初始神经网络（随机参数）的准确率
# %%
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 0] = np.mean(predicted_labels == Labels_test[0:5000])
print('\n 初始参数在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[0, 0] * 100) + '%')

# 场景 2（白天）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 1] = np.mean(predicted_labels == Labels_test[5074:10074])
print('\n 初始参数在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[0, 1] * 100) + '%')

# 场景 2（夜晚）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 2] = np.mean(predicted_labels == Labels_test[10074:15074])
print('\n 初始参数在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[0, 2] * 100) + '%')

# 场景 2（干扰）的准确率评估
hidden_1_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_1_weights) + hidden_1_biases)
hidden_2_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 1], hidden_2_weights) + hidden_2_biases)
hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 3] = np.mean(predicted_labels == Labels_test[15074:20074])
print('\n 初始参数在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[0, 3] * 100) + '%')
# %%

# 训练神经网络
for epoch in range(num_epochs):

    print('\n 开始第 #' + str(epoch) + ' 轮训练')
    # 打乱训练集
    perm = np.random.permutation(len(Labels_OG))
    Labels_train = Labels_OG[perm]
    Input_neurons = Input_neurons_OG[perm]

    # 在小批量数据上训练
    print('\n 小批量数据训练中... \n')
    for i in tqdm(range(num_batches)):
        # 选择一个小批量
        start = i * batch_size
        end = start + batch_size
        batch_images = Input_neurons[start:end, :, :]
        batch_labels = Labels_train[start:end]

        # 前向传播
        hidden_1_activations = sigmoid(np.dot(batch_images[:, :, 0], hidden_1_weights) + hidden_1_biases)
        hidden_2_activations = sigmoid(np.dot(batch_images[:, :, 1], hidden_2_weights) + hidden_2_biases)
        hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
        output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)

        # 反向传播：计算损失函数的梯度
        output_grads = output_activations
        output_grads[np.arange(len(batch_labels)), batch_labels] -= 1
        output_grads /= len(batch_labels)
        # 反向传播：计算隐藏层的梯度
        hidden_grads = np.dot(output_grads, output_weights.T) * hidden_activations * (1 - hidden_activations)
        # 分为两个独立的部分
        hidden_1_grads = hidden_grads[:, 0:hidden_1_size]
        hidden_2_grads = hidden_grads[:, hidden_1_size:hidden_size]

        # 更新权重和偏置
        output_weights -= learning_rate * np.dot(hidden_activations.T, output_grads)
        output_biases -= learning_rate * np.sum(output_grads, axis=0, keepdims=True)
        hidden_1_weights -= learning_rate * np.dot(batch_images[:, :, 0].T, hidden_1_grads)
        hidden_2_weights -= learning_rate * np.dot(batch_images[:, :, 1].T, hidden_2_grads)
        hidden_1_biases -= learning_rate * np.sum(hidden_1_grads, axis=0, keepdims=True)
        hidden_2_biases -= learning_rate * np.sum(hidden_2_grads, axis=0, keepdims=True)

    # 评估模型在场景 1 测试集上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_1_labels'] = Labels_test[0:5000]
    predict_result['Sce_1_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 0] = np.mean(predict_result['Sce_1_labels'] == predict_result['Sce_1_predicts'])
    print('\n 模型在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[epoch + 1, 0] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（白天）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_day_labels'] = Labels_test[5074:10074]
    predict_result['Sce_2_day_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 1] = np.mean(predict_result['Sce_2_day_labels'] == predict_result['Sce_2_day_predicts'])
    print('\n 模型在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 1] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（夜晚）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_night_labels'] = Labels_test[10074:15074]
    predict_result['Sce_2_night_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 2] = np.mean(
        predict_result['Sce_2_night_labels'] == predict_result['Sce_2_night_predicts'])
    print('\n 模型在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 2] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')

    # 评估模型在场景 2 测试集（干扰）上的准确率
    hidden_1_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_1_weights) + hidden_1_biases)
    hidden_2_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 1], hidden_2_weights) + hidden_2_biases)
    hidden_activations = np.concatenate((hidden_1_activations, hidden_2_activations), axis=1)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_interference_labels'] = Labels_test[15074:20074]
    predict_result['Sce_2_interference_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 3] = np.mean(
        predict_result['Sce_2_interference_labels'] == predict_result['Sce_2_interference_predicts'])
    print('\n 模型在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 3] * 100) + '% (经过 #' + str(
        epoch + 1) + ' 轮训练)')
# %%

"""
导出模型
"""
# %%
Model_series = 'DBPD_mixed_Gen_5'
outdir_models = Model_r + '\\' + Model_series
if not os.path.exists(outdir_models):
    os.mkdir(outdir_models)
# 保存模型权重和偏置
np.save(outdir_models + '\output_weights.npy', output_weights)
np.save(outdir_models + '\output_biases.npy', output_biases)
np.save(outdir_models + '\hidden_1_weights.npy', hidden_1_weights)
np.save(outdir_models + '\hidden_2_weights.npy', hidden_2_weights)
np.save(outdir_models + '\hidden_1_biases.npy', hidden_1_biases)
np.save(outdir_models + '\hidden_2_biases.npy', hidden_2_biases)
accuracy.to_csv(outdir_models + '\\Accuracy_evolution.csv')

# 保存预测结果
outdir_results = rr + '\\Experiments' + Task_name + Task_subname
if not os.path.exists(outdir_results):
    os.mkdir(outdir_results)
predict_result.to_csv(outdir_results + '\\Predict_results_' + Model_series + '.csv')
accuracy.to_csv(outdir_results + '\\Accuracy_evolution_' + Model_series + '.csv')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
