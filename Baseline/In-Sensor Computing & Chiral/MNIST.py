import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os

# 加载并预处理MNIST数据集
(x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()

# 将图像归一化到0到1之间
x_train = x_train / 255.0
x_test_original = x_test_original / 255.0

# 重塑图像数据，添加通道维度（因为卷积神经网络需要输入图像有4维：batch_size, height, width, channels）
x_train = x_train.reshape(-1, 28, 28, 1)
x_test_original = x_test_original.reshape(-1, 28, 28, 1)

# 设备参数（这些值用于生成混合数据集）
R_form_LH = 0.962
R_form_RH = 0.0383
S_form_LH = 0.0492
S_form_RH = 0.951
intensity_LH = 0.5
intensity_RH = 1 - intensity_LH

# 创建混合数据集的函数，基于设备参数生成带标签的混合图像
def create_device_mixed_data(x_data, y_data, LH_param, RH_param, label_type='label1'):
    n = x_data.shape[0]  # 获取数据集的大小
    mixed_images = []  # 用于存储混合后的图像
    mixed_labels = []  # 用于存储混合后的标签

    # 生成混合数据（按顺序将每两张图像混合）
    for i in range(n - 1):
        img1, label1 = x_data[i], y_data[i]
        img2, label2 = x_data[i + 1], y_data[i + 1]
        mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH  # 混合两张图像
        mixed_img = mixed_img / np.max(mixed_img)  # 归一化到0到1之间
        mixed_images.append(mixed_img)
        # 根据标签类型决定选择哪个标签
        mixed_labels.append(label1 if label_type == 'label1' else label2)

    # 处理最后一张图像（与第一张图像混合）
    img1, label1 = x_data[-1], y_data[-1]
    img2, label2 = x_data[0], y_data[0]
    mixed_img = img1 * LH_param * intensity_LH + img2 * RH_param * intensity_RH
    mixed_img = mixed_img / np.max(mixed_img)
    mixed_images.append(mixed_img)
    mixed_labels.append(label1 if label_type == 'label1' else label2)

    return np.array(mixed_images), np.array(mixed_labels)

# 使用设备参数创建R-form和S-form数据集
x_test_R_form, y_test_R_form = create_device_mixed_data(
    x_test_original, y_test_original, R_form_LH, R_form_RH, label_type='label1')
x_test_S_form, y_test_S_form = create_device_mixed_data(
    x_test_original, y_test_original, S_form_LH, S_form_RH, label_type='label2')

# 创建不使用设备参数的混合数据集（用于比较）
x_test_mixed, y_test_mixed = create_device_mixed_data(
    x_test_original, y_test_original, 1.0, 1.0, label_type='label1')

# 准备类别标签，使用categorical_crossentropy损失函数
y_train_cat = to_categorical(y_train, 10)
y_test_original_cat = to_categorical(y_test_original, 10)

# 定义卷积神经网络（CNN）模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 输出10个类别
])

# 编译模型，使用Adam优化器，学习率设置为0.000002
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 数据增强设置，旋转、缩放、平移
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

batch_size = 64  # 批量大小
epochs = 20  # 训练20个epoch
train_gen = datagen.flow(x_train, y_train_cat, batch_size=batch_size)
test_gen = datagen.flow(x_test_original, y_test_original_cat, batch_size=batch_size)

# 为结果创建目录（所有结果将保存到'结果'文件夹中）
os.makedirs('results', exist_ok=True)

# 初始化数组用于存储每个epoch的准确率
accuracy_data = np.empty((epochs, 5))

# 训练和评估模型
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(train_gen, epochs=1, validation_data=test_gen, verbose=1)

    # 在原始测试集上评估
    pred_test = np.argmax(model.predict(x_test_original), axis=-1)
    test_acc = np.mean(pred_test == y_test_original.flatten())

    # 在混合数据集上评估
    pred_mixed = np.argmax(model.predict(x_test_mixed), axis=-1)
    test_acc_mixed = np.mean(pred_mixed == y_test_mixed.flatten())

    # 在R-form数据集上评估
    pred_R_form = np.argmax(model.predict(x_test_R_form), axis=-1)
    test_acc_R_form = np.mean(pred_R_form == y_test_R_form.flatten())

    # 在S-form数据集上评估
    pred_S_form = np.argmax(model.predict(x_test_S_form), axis=-1)
    test_acc_S_form = np.mean(pred_S_form == y_test_S_form.flatten())

    # 存储准确率数据：每个epoch对应[epoch, 原始测试集, 混合数据集, R-form数据集, S-form数据集]
    accuracy_data[epoch] = [epoch + 1, test_acc, test_acc_mixed, test_acc_R_form, test_acc_S_form]

    # 打印每个数据集的准确率
    print(f"Test Accuracy (Original): {test_acc:.4f}")
    print(f"Test Accuracy (Mixed): {test_acc_mixed:.4f}")
    print(f"Test Accuracy (R-form): {test_acc_R_form:.4f}")
    print(f"Test Accuracy (S-form): {test_acc_S_form:.4f}")

# 将准确率数据保存为CSV格式
np.savetxt('results/MNIST_accuracy_data.csv', accuracy_data, fmt='%1.4f', delimiter=',')

# 保存混淆矩阵为热力图和数值数据的函数
def save_confusion_matrix(y_true, y_pred, filename_image, filename_data):
    cm = confusion_matrix(y_true, y_pred)

    # 保存混淆矩阵为热力图
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, square=True, annot=False, fmt='d', cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename_image)
    plt.close()  # 显式关闭图形以释放资源

    # 保存混淆矩阵为数值数据（CSV）
    np.savetxt(filename_data, cm, fmt='%d', delimiter=',')

# 使用最后一次epoch的预测保存混淆矩阵
save_confusion_matrix(y_test_original.flatten(), pred_test, 'results/MNIST_confusion_matrix_original.png',
                      'results/MNIST_confusion_matrix_original.csv')
save_confusion_matrix(y_test_mixed.flatten(), pred_mixed, 'results/MNIST_confusion_matrix_mixed.png',
                      'resultsMNIST_confusion_matrix_mixed.csv')
save_confusion_matrix(y_test_R_form.flatten(), pred_R_form, 'resultsMNIST_confusion_matrix_R_form.png',
                      'resultsMNIST_confusion_matrix_R_form.csv')
save_confusion_matrix(y_test_S_form.flatten(), pred_S_form, 'results/MNIST_confusion_matrix_S_form.png',
                      'results/MNIST_confusion_matrix_S_form.csv')

