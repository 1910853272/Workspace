<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
ANN_architect belongs to DBPD simulation
Created on April 05

@author: Wang Jialiang
"""

"""
导入所需的包
"""
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from __Mod_route__ import get_root

# 获取存储模型和结果的根目录
rr = get_root()
Task_name = '\20240927'
Task_subname = '_mixed_dataset_recog'

"""
函数定义
"""


def sigmoid(x):
    # Sigmoid 激活函数
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # Softmax 激活函数，将 logits 转换为概率
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def norm(x):
    # 归一化的占位符函数（当前未使用）
    for i in x.shape[2]:
        for j in x.shape[1]:
            pass
    return


"""
加载输入数据
"""
# %%
# 加载输入数据的占位符（已注释）
# Model_r = rr + '\Models' + Task_name + Task_subname
# Input_neurons_OG = np.load(Model_r + '\Train_set_DBPD_Input_neurons.npy')
# Labels_OG = np.load(Model_r + '\Train_set_Labels.npy')
# Input_neurons_test = np.load(Model_r + '\Test_set_DBPD_Input_neurons.npy')
# Labels_test = np.load(Model_r + '\Test_set_Labels.npy')

# Labels_OG = Labels_OG.astype(np.int64)
# Labels_test = Labels_test.astype(np.int64)

"""
神经网络框架
"""
# %%
# 定义神经网络架构
input_size = 50 * 50  # 输入层的大小
hidden_size = 300  # 隐藏层的神经元数量
output_size = 3  # 输出类别的数量
learning_rate = 0.01  # 梯度下降的学习率

# 随机初始化隐藏层和输出层的权重和偏置
hidden_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
hidden_weights = hidden_weights / 10  # 缩放以平衡较大的输入值
hidden_biases = np.zeros((1, hidden_size))
output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
output_biases = np.zeros((1, output_size))

# 加载预训练模型权重的占位符（已注释）
# Model_series = '\NBPD200_mixed_Gen_5'
# hidden_weights = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\hidden_weights.npy')
# hidden_biases = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\hidden_biases.npy')
# output_weights = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\output_weights.npy')
# output_biases = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\output_biases.npy')

"""
神经网络构建
"""
# %%
# 设置训练参数
num_epochs = 20  # 训练模型的迭代次数
batch_size = 180  # 每个小批量的大小
num_batches = len(Labels_OG) // batch_size  # 每个 epoch 中的小批量数量

# 初始化用于跟踪不同场景下准确率的 DataFrame
accuracy = pd.DataFrame(index=range(num_epochs + 1),
                        columns=['Sce_1', 'Sce_2_day', 'Sce_2_night', 'Sce_2_interference'], dtype='float64')

# 初始化用于存储预测结果的 DataFrame
predict_result = pd.DataFrame(index=range(5000),
                              columns=['Sce_1_labels', 'Sce_1_predicts',
                                       'Sce_2_day_labels', 'Sce_2_day_predicts',
                                       'Sce_2_night_labels', 'Sce_2_night_predicts',
                                       'Sce_2_interference_labels', 'Sce_2_interference_predicts'])

# 评估初始神经网络的准确率（随机参数）
hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 0] = np.mean(predicted_labels == Labels_test[0:5000])
print('\n 初始参数在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[0, 0] * 100) + '%')

# 针对不同场景重复评估（白天、夜晚、干扰）
# 场景 2（白天）
hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 1] = np.mean(predicted_labels == Labels_test[5074:10074])
print('\n 初始参数在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[0, 1] * 100) + '%')

# 场景 2（夜晚）
hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 2] = np.mean(predicted_labels == Labels_test[10074:15074])
print('\n 初始参数在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[0, 2] * 100) + '%')

# 场景 2（干扰）
hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 3] = np.mean(predicted_labels == Labels_test[15074:20074])
print('\n 初始参数在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[0, 3] * 100) + '%')

# 训练循环
for epoch in range(num_epochs):
    print('\n 开始第 #' + str(epoch) + ' 轮训练')
    # 打乱训练集
    perm = np.random.permutation(len(Labels_OG))
    Labels_train = Labels_OG[perm]
    Input_neurons = Input_neurons_OG[perm]

    # 使用小批量进行训练
    print('\n 小批量训练进行中... \n')
    for i in tqdm(range(num_batches)):
        # 选择一个小批量
        start = i * batch_size
        end = start + batch_size
        batch_images = Input_neurons[start:end, :, 0]
        batch_labels = Labels_train[start:end]

        # 前向传播
        hidden_activations = sigmoid(np.dot(batch_images, hidden_weights) + hidden_biases)
        output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)

        # 反向传播：计算输出层的梯度
        output_grads = output_activations
        output_grads[np.arange(len(batch_labels)), batch_labels] -= 1  # 计算损失对 logits 的梯度
        output_grads /= len(batch_labels)
        # 反向传播：计算隐藏层的梯度
        hidden_grads = np.dot(output_grads, output_weights.T) * hidden_activations * (
                    1 - hidden_activations)  # 计算损失对隐藏层的梯度

        # 梯度下降：更新权重和偏置
        output_weights -= learning_rate * np.dot(hidden_activations.T, output_grads)
        output_biases -= learning_rate * np.sum(output_grads, axis=0, keepdims=True)
        hidden_weights -= learning_rate * np.dot(batch_images.T, hidden_grads)
        hidden_biases -= learning_rate * np.sum(hidden_grads, axis=0, keepdims=True)

    # 每个 epoch 后评估不同场景下的准确率
    # 场景 1
    hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_1_labels'] = Labels_test[0:5000]
    predict_result['Sce_1_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 0] = np.mean(predict_result['Sce_1_labels'] == predict_result['Sce_1_predicts'])
    print('\n 模型在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[epoch + 1, 0] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（白天）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_day_labels'] = Labels_test[5074:10074]
    predict_result['Sce_2_day_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 1] = np.mean(predict_result['Sce_2_day_labels'] == predict_result['Sce_2_day_predicts'])
    print('\n 模型在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 1] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（夜晚）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_night_labels'] = Labels_test[10074:15074]
    predict_result['Sce_2_night_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 2] = np.mean(
        predict_result['Sce_2_night_labels'] == predict_result['Sce_2_night_predicts'])
    print('\n 模型在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 2] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（干扰）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_interference_labels'] = Labels_test[15074:20074]
    predict_result['Sce_2_interference_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 3] = np.mean(
        predict_result['Sce_2_interference_labels'] == predict_result['Sce_2_interference_predicts'])
    print('\n 模型在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 3] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

"""
导出模型
"""
# %%
# 导出模型和结果的占位符（已注释）
# Model_series = 'NBPD200_mixed_Gen_1'
# outdir_models = Model_r + '\' + Model_series
# if not os.path.exists(outdir_models):
#     os.mkdir(outdir_models)
# np.save(outdir_models + '\output_weights.npy', output_weights)
# np.save(outdir_models + '\output_biases.npy', output_biases)
# np.save(outdir_models + '\hidden_weights.npy', hidden_weights)
# np.save(outdir_models + '\hidden_biases.npy', hidden_biases)
# accuracy.to_csv(outdir_models + '\Accuracy_evolution.csv')

# outdir_results = rr + '\Experiments' + Task_name + Task_subname
# if not os.path.exists(outdir_results):
#     os.mkdir(outdir_results)
# predict_result.to_csv(outdir_results + '\Predict_results_' + Model_series + '.csv')
# accuracy.to_csv(outdir_results + '\Accuracy_evolution_' + Model_series + '.csv')
=======
# -*- coding: utf-8 -*-
"""
ANN_architect belongs to DBPD simulation
Created on April 05

@author: Wang Jialiang
"""

"""
导入所需的包
"""
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from __Mod_route__ import get_root

# 获取存储模型和结果的根目录
rr = get_root()
Task_name = '\20240927'
Task_subname = '_mixed_dataset_recog'

"""
函数定义
"""


def sigmoid(x):
    # Sigmoid 激活函数
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # Softmax 激活函数，将 logits 转换为概率
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def norm(x):
    # 归一化的占位符函数（当前未使用）
    for i in x.shape[2]:
        for j in x.shape[1]:
            pass
    return


"""
加载输入数据
"""
# %%
# 加载输入数据的占位符（已注释）
# Model_r = rr + '\Models' + Task_name + Task_subname
# Input_neurons_OG = np.load(Model_r + '\Train_set_DBPD_Input_neurons.npy')
# Labels_OG = np.load(Model_r + '\Train_set_Labels.npy')
# Input_neurons_test = np.load(Model_r + '\Test_set_DBPD_Input_neurons.npy')
# Labels_test = np.load(Model_r + '\Test_set_Labels.npy')

# Labels_OG = Labels_OG.astype(np.int64)
# Labels_test = Labels_test.astype(np.int64)

"""
神经网络框架
"""
# %%
# 定义神经网络架构
input_size = 50 * 50  # 输入层的大小
hidden_size = 300  # 隐藏层的神经元数量
output_size = 3  # 输出类别的数量
learning_rate = 0.01  # 梯度下降的学习率

# 随机初始化隐藏层和输出层的权重和偏置
hidden_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
hidden_weights = hidden_weights / 10  # 缩放以平衡较大的输入值
hidden_biases = np.zeros((1, hidden_size))
output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
output_biases = np.zeros((1, output_size))

# 加载预训练模型权重的占位符（已注释）
# Model_series = '\NBPD200_mixed_Gen_5'
# hidden_weights = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\hidden_weights.npy')
# hidden_biases = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\hidden_biases.npy')
# output_weights = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\output_weights.npy')
# output_biases = np.load(rr + '\Models\' + Task_name + Task_subname + Model_series + '\output_biases.npy')

"""
神经网络构建
"""
# %%
# 设置训练参数
num_epochs = 20  # 训练模型的迭代次数
batch_size = 180  # 每个小批量的大小
num_batches = len(Labels_OG) // batch_size  # 每个 epoch 中的小批量数量

# 初始化用于跟踪不同场景下准确率的 DataFrame
accuracy = pd.DataFrame(index=range(num_epochs + 1),
                        columns=['Sce_1', 'Sce_2_day', 'Sce_2_night', 'Sce_2_interference'], dtype='float64')

# 初始化用于存储预测结果的 DataFrame
predict_result = pd.DataFrame(index=range(5000),
                              columns=['Sce_1_labels', 'Sce_1_predicts',
                                       'Sce_2_day_labels', 'Sce_2_day_predicts',
                                       'Sce_2_night_labels', 'Sce_2_night_predicts',
                                       'Sce_2_interference_labels', 'Sce_2_interference_predicts'])

# 评估初始神经网络的准确率（随机参数）
hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 0] = np.mean(predicted_labels == Labels_test[0:5000])
print('\n 初始参数在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[0, 0] * 100) + '%')

# 针对不同场景重复评估（白天、夜晚、干扰）
# 场景 2（白天）
hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 1] = np.mean(predicted_labels == Labels_test[5074:10074])
print('\n 初始参数在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[0, 1] * 100) + '%')

# 场景 2（夜晚）
hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 2] = np.mean(predicted_labels == Labels_test[10074:15074])
print('\n 初始参数在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[0, 2] * 100) + '%')

# 场景 2（干扰）
hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_weights) + hidden_biases)
output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
predicted_labels = np.argmax(output_activations, axis=1)
accuracy.iloc[0, 3] = np.mean(predicted_labels == Labels_test[15074:20074])
print('\n 初始参数在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[0, 3] * 100) + '%')

# 训练循环
for epoch in range(num_epochs):
    print('\n 开始第 #' + str(epoch) + ' 轮训练')
    # 打乱训练集
    perm = np.random.permutation(len(Labels_OG))
    Labels_train = Labels_OG[perm]
    Input_neurons = Input_neurons_OG[perm]

    # 使用小批量进行训练
    print('\n 小批量训练进行中... \n')
    for i in tqdm(range(num_batches)):
        # 选择一个小批量
        start = i * batch_size
        end = start + batch_size
        batch_images = Input_neurons[start:end, :, 0]
        batch_labels = Labels_train[start:end]

        # 前向传播
        hidden_activations = sigmoid(np.dot(batch_images, hidden_weights) + hidden_biases)
        output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)

        # 反向传播：计算输出层的梯度
        output_grads = output_activations
        output_grads[np.arange(len(batch_labels)), batch_labels] -= 1  # 计算损失对 logits 的梯度
        output_grads /= len(batch_labels)
        # 反向传播：计算隐藏层的梯度
        hidden_grads = np.dot(output_grads, output_weights.T) * hidden_activations * (
                    1 - hidden_activations)  # 计算损失对隐藏层的梯度

        # 梯度下降：更新权重和偏置
        output_weights -= learning_rate * np.dot(hidden_activations.T, output_grads)
        output_biases -= learning_rate * np.sum(output_grads, axis=0, keepdims=True)
        hidden_weights -= learning_rate * np.dot(batch_images.T, hidden_grads)
        hidden_biases -= learning_rate * np.sum(hidden_grads, axis=0, keepdims=True)

    # 每个 epoch 后评估不同场景下的准确率
    # 场景 1
    hidden_activations = sigmoid(np.dot(Input_neurons_test[0:5000, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_1_labels'] = Labels_test[0:5000]
    predict_result['Sce_1_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 0] = np.mean(predict_result['Sce_1_labels'] == predict_result['Sce_1_predicts'])
    print('\n 模型在场景 1 测试集上的准确率为 ' + str(accuracy.iloc[epoch + 1, 0] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（白天）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[5074:10074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_day_labels'] = Labels_test[5074:10074]
    predict_result['Sce_2_day_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 1] = np.mean(predict_result['Sce_2_day_labels'] == predict_result['Sce_2_day_predicts'])
    print('\n 模型在场景 2 测试集（白天）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 1] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（夜晚）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[10074:15074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_night_labels'] = Labels_test[10074:15074]
    predict_result['Sce_2_night_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 2] = np.mean(
        predict_result['Sce_2_night_labels'] == predict_result['Sce_2_night_predicts'])
    print('\n 模型在场景 2 测试集（夜晚）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 2] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

    # 场景 2（干扰）
    hidden_activations = sigmoid(np.dot(Input_neurons_test[15074:20074, :, 0], hidden_weights) + hidden_biases)
    output_activations = softmax(np.dot(hidden_activations, output_weights) + output_biases)
    predicted_labels = np.argmax(output_activations, axis=1)
    predict_result['Sce_2_interference_labels'] = Labels_test[15074:20074]
    predict_result['Sce_2_interference_predicts'] = predicted_labels
    accuracy.iloc[epoch + 1, 3] = np.mean(
        predict_result['Sce_2_interference_labels'] == predict_result['Sce_2_interference_predicts'])
    print('\n 模型在场景 2 测试集（干扰）上的准确率为 ' + str(accuracy.iloc[epoch + 1, 3] * 100) + '%，经过 #' + str(
        epoch + 1) + ' 轮训练后')

"""
导出模型
"""
# %%
# 导出模型和结果的占位符（已注释）
# Model_series = 'NBPD200_mixed_Gen_1'
# outdir_models = Model_r + '\' + Model_series
# if not os.path.exists(outdir_models):
#     os.mkdir(outdir_models)
# np.save(outdir_models + '\output_weights.npy', output_weights)
# np.save(outdir_models + '\output_biases.npy', output_biases)
# np.save(outdir_models + '\hidden_weights.npy', hidden_weights)
# np.save(outdir_models + '\hidden_biases.npy', hidden_biases)
# accuracy.to_csv(outdir_models + '\Accuracy_evolution.csv')

# outdir_results = rr + '\Experiments' + Task_name + Task_subname
# if not os.path.exists(outdir_results):
#     os.mkdir(outdir_results)
# predict_result.to_csv(outdir_results + '\Predict_results_' + Model_series + '.csv')
# accuracy.to_csv(outdir_results + '\Accuracy_evolution_' + Model_series + '.csv')
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
