import numpy as np
import pandas as pd

class SingleLayerPerceptron:
    def __init__(self, input_size, output_size):
        """初始化感知器的权重和偏置，使用小的随机值进行初始化"""
        # 随机生成权重矩阵，大小为 (输出大小, 输入大小)，并乘以0.1以便值不太大
        self.weights = np.random.randn(output_size, input_size) * 0.1
        # 随机生成偏置向量，大小为 (输出大小)，并乘以0.1
        self.biases = np.random.randn(output_size) * 0.1

def memristor_quantize(values, bits=4):
    """使用线性忆阻器模型对值进行量化，指定量化位数"""
    # 计算量化级别，即最大值和最小值之间的不同层次数
    levels = 2 ** bits
    # 获取输入值的最大值和最小值
    max_val = np.max(values)
    min_val = np.min(values)
    # 计算量化步长，步长决定了值的离散化程度
    step = (max_val - min_val) / (levels - 1)
    # 对值进行量化：通过除以步长并四舍五入，得到离散化后的值
    quantized = np.round((values - min_val) / step) * step + min_val
    return quantized

def ann_to_snn(weights, biases, v_th=1.0):
    """将ANN（人工神经网络）的权重和偏置量化为SNN（脉冲神经网络）的权重和偏置"""
    # 计算一个比例因子，将权重和偏置缩放，使其在SNN中不超过设定的阈值
    scale_factor = np.max(np.abs(weights)) / v_th
    # 将ANN的权重和偏置按比例缩放
    weights_scaled = weights / scale_factor
    biases_scaled = biases / scale_factor
    # 对权重和偏置进行量化
    weights_quantized = memristor_quantize(weights_scaled, 4)  # 使用4位量化
    biases_quantized = memristor_quantize(biases_scaled, 4)  # 使用4位量化
    return weights_quantized, biases_quantized

def snn_inference(weights, biases, input_frequencies, time_steps=100, dt=1.0):
    """在SNN中进行推理，给定输入频率、权重和偏置"""
    # 获取输入和输出的大小
    input_size = weights.shape[1]  # 输入的特征数量
    output_size = weights.shape[0]  # 输出的神经元数量
    # 初始化脉冲响应数组
    spike_response = np.zeros(output_size)
    # 模拟SNN的推理过程
    for _ in range(time_steps):
        # 根据输入频率生成脉冲：随机生成一个数组，将小于频率的值设置为1，表示神经元发放脉冲
        inputs = np.random.rand(input_size) < (input_frequencies * dt)
        # 计算膜电位：根据输入脉冲和权重进行计算，加入偏置
        membrane_potentials = np.dot(weights, inputs) + biases
        # 输出脉冲：当膜电位大于等于阈值（1.0）时，神经元发放脉冲
        outputs = membrane_potentials >= 1.0
        # 更新脉冲响应
        spike_response += outputs
        # 更新偏置：每次输出脉冲后，偏置减少，模拟神经元的疲劳
        biases -= outputs * 1.0
    return spike_response

def compute_accuracy(predictions, targets):
    """计算预测值与目标值之间的准确率"""
    # 计算预测值和目标值相等的数量
    correct = np.sum(predictions == targets)
    # 计算准确率：正确预测的比例
    return correct / len(predictions)

def load_and_process_data(filepath):
    """加载和处理CSV数据，基于电压阈值计算输入频率"""
    # 读取CSV文件
    df = pd.read_csv(filepath)
    threshold = 2.0  # 设定电压阈值为2.0伏特
    # 对每一列数据进行处理，计算大于阈值的元素个数，即表示该输入的频率
    input_frequencies = (df > threshold).sum(axis=0)  # 对每列数据求和，得到大于阈值的数量
    return input_frequencies

# 加载并处理数据
filepath = '/opticneuronspikes/Sensoryspikes.csv'
input_frequencies = load_and_process_data(filepath)

# 定义网络的输入和输出大小
input_size = 20  # 输入大小为20
output_size = 4  # 输出大小为4
perceptron = SingleLayerPerceptron(input_size, output_size)  # 创建一个单层感知器对象
# 将ANN的权重和偏置量化为SNN的权重和偏置
snn_weights, snn_biases = ann_to_snn(perceptron.weights, perceptron.biases)

# 生成理想输出（用于测试）
ideal_outputs = (input_frequencies > 50).astype(int)[:output_size]  # 将输入频率大于50的视为1，否则为0

# 执行SNN推理，得到输出脉冲
output_spikes = snn_inference(snn_weights, snn_biases, input_frequencies.values)
time_steps = 20  # 设置时间步数

# 计算推理的准确率：如果输出脉冲大于一半的时间步，则认为是一个正确的输出
accuracy = compute_accuracy(output_spikes > (time_steps // 2), ideal_outputs)
print("Output spikes:", output_spikes)
print("Ideal outputs:", ideal_outputs)
print("Inference Accuracy: {:.2f}%".format(accuracy * 100))
