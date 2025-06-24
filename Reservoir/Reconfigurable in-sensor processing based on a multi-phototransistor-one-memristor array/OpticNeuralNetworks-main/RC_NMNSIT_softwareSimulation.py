import os
import numpy as np
import cv2
import easyesn
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot


def load_nmnist_data(path):
    """
    从指定路径加载N-MNIST数据集。
    假设数据集存储在'训练'和'测试'文件夹中，每个类别有一个子文件夹。

    参数:
    - path: 数据集的根路径。

    返回:
    - x_train: 训练集图像数据。
    - y_train: 训练集标签。
    - x_test: 测试集图像数据。
    - y_test: 测试集标签。
    """

    def load_images_from_folder(folder):
        images = []  # 存储图像数据
        labels = []  # 存储对应的标签
        # 遍历文件夹中的每个类别子文件夹，按字母顺序排序
        for class_idx, class_folder in enumerate(sorted(os.listdir(folder))):
            class_folder_path = os.path.join(folder, class_folder)
            # 遍历每个类别文件夹中的所有图像文件，按字母顺序排序
            for filename in sorted(os.listdir(class_folder_path)):
                img_path = os.path.join(class_folder_path, filename)
                # 读取灰度图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)  # 添加图像数据
                    labels.append(class_idx)  # 添加对应的标签（类别索引）
        return np.array(images), np.array(labels)

    # 加载训练集和测试集的数据和标签
    x_train, y_train = load_images_from_folder(os.path.join(path, 'train'))
    x_test, y_test = load_images_from_folder(os.path.join(path, 'test'))

    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, x_test, target_size=(20, 20), time_steps=10):
    """
    对数据进行预处理，包括调整大小和重新形状。

    参数:
    - x_train: 原始训练集图像数据。
    - x_test: 原始测试集图像数据。
    - target_size: 调整后的图像大小（宽，高）。
    - time_steps: 时间步长，用于序列数据的处理。

    返回:
    - x_train_reshaped: 预处理后的训练集数据。
    - x_test_reshaped: 预处理后的测试集数据。
    """
    # 调整训练集和测试集图像的大小
    x_train_resized = np.array([cv2.resize(img, target_size) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, target_size) for img in x_test])

    # 重新形状，使其适应后续模型的输入需求
    # 假设每个样本有多个时间步的数据
    x_train_reshaped = x_train_resized.reshape(-1, time_steps, target_size[0], target_size[1])
    x_test_reshaped = x_test_resized.reshape(-1, time_steps, target_size[0], target_size[1])

    return x_train_reshaped, x_test_reshaped


def create_reservoir_layer(input_shape, n_reservoir=20, spectral_radius=0.9, noise_level=0.01):
    """
    使用easyesn创建一个回声状态网络（Echo State Network，ESN）作为Reservoir Computing层。

    参数:
    - input_shape: 输入数据的形状。
    - n_reservoir: Reservoir的神经元数量。
    - spectral_radius: 储层权重矩阵的谱半径，用于控制动态行为。
    - noise_level: 注入到网络中的噪声水平，有助于网络的泛化能力。

    返回:
    - esn: 创建的ESN对象。
    """
    esn = easyesn.ESN(
        n_input=input_shape[1] * input_shape[2],  # 输入层神经元数，等于每个时间步的输入特征数
        n_output=n_reservoir,  # 输出层神经元数，即Reservoir的大小
        n_reservoir=n_reservoir,  # Reservoir的神经元数量
        spectralRadius=spectral_radius,  # 储层权重矩阵的谱半径
        noiseLevel=noise_level  # 注入的噪声水平
    )
    return esn


def transform_data_with_esn(esn, data):
    """
    使用回声状态网络（ESN）转换数据。

    参数:
    - esn: 已创建并训练好的ESN对象。
    - data: 输入数据，形状为（样本数， 时间步长， 高度， 宽度）。

    返回:
    - transformed_data: 通过ESN转换后的数据，形状为（样本数， Reservoir大小）。
    """
    transformed_data = []  # 存储转换后的数据
    for sequence in data:
        sequence = sequence.reshape(sequence.shape[0], -1)  # 将每个时间步的数据展平
        esn_output = esn.simulate(sequence)  # 通过ESN进行仿真
        transformed_data.append(esn_output[-1])  # 使用Reservoir的最后一个状态作为特征
    return np.array(transformed_data)


def plot_samples(x, y, num_samples=10):
    """
    绘制数据集中样本图像。

    参数:
    - x: 图像数据，形状为（样本数， 特征数）。
    - y: 标签数据，形状为（样本数，）。
    - num_samples: 要绘制的样本数量。
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x[i].reshape(20, 20), cmap='gray')  # 假设每个图像被调整为20x20大小
        plt.title(f"Label: {y[i]}")  # 显示标签
        plt.axis('off')  # 不显示坐标轴
    plt.show()


def plot_weights(model):
    """
    绘制模型各层的权重分布。

    参数:
    - model: 已训练好的Keras模型。
    """
    for layer in model.layers:
        if 'dense' in layer.name:  # 仅绘制全连接层的权重
            weights, biases = layer.get_weights()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f'{layer.name} weights')
            plt.hist(weights.flatten(), bins=50)  # 绘制权重直方图
            plt.subplot(1, 2, 2)
            plt.title(f'{layer.name} biases')
            plt.hist(biases.flatten(), bins=50)  # 绘制偏置直方图
            plt.show()


def create_classification_model(input_shape):
    """
    创建一个简单的分类模型。

    参数:
    - input_shape: 输入特征的形状（即Reservoir的大小）。

    返回:
    - model: 构建好的Keras分类模型。
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))  # 输入层
    model.add(layers.Dense(10, activation='softmax'))  # 输出层，10个类别，使用softmax激活函数
    return model


# ================================
# 主程序开始
# ================================

# 加载和预处理数据
x_train, y_train, x_test, y_test = load_nmnist_data('./n_mnist')  # 加载N-MNIST数据集
x_train, x_test = preprocess_data(x_train, x_test)  # 对数据进行预处理（调整大小和重新形状）

# 绘制数据集中的样本图像
plot_samples(x_train[:, 0], y_train, num_samples=10)  # 绘制训练集的前10个样本

# 对标签进行One-Hot编码
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))  # 对训练集标签进行编码
y_test = encoder.transform(y_test.reshape(-1, 1))  # 对测试集标签进行编码

# 创建并训练ESN
esn = create_reservoir_layer(x_train.shape, n_reservoir=20, spectral_radius=1.2, noise_level=0.05)  # 创建ESN对象

# 使用ESN转换数据
x_train_transformed = transform_data_with_esn(esn, x_train)  # 转换训练集数据
x_test_transformed = transform_data_with_esn(esn, x_test)  # 转换测试集数据

# 创建并编译分类模型
model = create_classification_model(x_train_transformed.shape[1])  # 创建分类模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型

# 使用验证集训练模型
history = model.fit(
    x_train_transformed, y_train,
    epochs=200,  # 训练200个epoch
    batch_size=64,  # 每个批次64个样本
    validation_split=0.2  # 使用20%的训练数据作为验证集
)

# 评估模型在测试集上的性能
loss, accuracy = model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy: {accuracy}')  # 输出测试集上的准确率

# 生成分类报告
y_pred = model.predict(x_test_transformed)  # 获取模型对测试集的预测
y_pred_classes = np.argmax(y_pred, axis=1)  # 获取预测的类别
y_true_classes = np.argmax(y_test, axis=1)  # 获取真实的类别
print(classification_report(y_true_classes, y_pred_classes))  # 输出分类报告

# 绘制训练和验证的准确率及损失曲线
plt.figure(figsize=(12, 6))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 训练准确率
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 验证准确率
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')  # 训练损失
plt.plot(history.history['val_loss'], label='Validation Loss')  # 验证损失
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 绘制模型各层的权重分布
plot_weights(model)


# ================================
# 模型量化部分
# ================================

def apply_4bit_quantization(model):
    """
    对模型应用4位量化，使用TensorFlow Model Optimization Toolkit。

    参数:
    - model: 原始的Keras模型。

    返回:
    - q_model: 量化后的Keras模型。
    """
    # 定义量化模型的策略，这里使用默认的8位量化方案作为示例
    quantize_model = tfmot.quantization.keras.quantize_model
    q_model = quantize_model(model, tfmot.experimental.combine.Default8BitQuantizeScheme())

    # 重新编译量化后的模型
    q_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return q_model


# 对原始模型进行量化
q_model = apply_4bit_quantization(model)  # 创建量化后的模型

# 使用验证集继续训练量化后的模型
q_history = q_model.fit(
    x_train_transformed, y_train,
    epochs=200,  # 继续训练200个epoch
    batch_size=64,  # 每个批次64个样本
    validation_split=0.2  # 使用20%的训练数据作为验证集
)

# 评估量化后的模型在测试集上的性能
q_loss, q_accuracy = q_model.evaluate(x_test_transformed, y_test)
print(f'Test accuracy after quantization: {q_accuracy}')  # 输出量化后模型在测试集上的准确率

# 绘制量化后模型的训练和验证准确率及损失曲线
plt.figure(figsize=(12, 6))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(q_history.history['accuracy'], label='Train Accuracy')  # 训练准确率
plt.plot(q_history.history['val_accuracy'], label='Validation Accuracy')  # 验证准确率
plt.title('Quantized Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(q_history.history['loss'], label='Train Loss')  # 训练损失
plt.plot(q_history.history['val_loss'], label='Validation Loss')  # 验证损失
plt.title('Quantized Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
