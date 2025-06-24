import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

# 自定义一个回调类，用于记录每个训练批次的准确率
class BatchAccuracyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_accuracies = []  # 初始化存储批次准确率的列表

    def on_train_batch_end(self, batch, logs=None):
        self.batch_accuracies.append(logs.get('accuracy'))  # 每个批次结束时记录准确率

# 自定义权重约束函数，将权重值限制在0和1之间
def weight_constraint(w):
    return K.clip(w, 0, 1)

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
# 将训练图像和测试图像重新调整为20x20的尺寸并标准化
train_images = train_images.reshape((60000, 28, 28))  # 调整为28x28的二维图像
train_images = np.asarray([np.asarray(Image.fromarray(image).resize((20, 20))) for image in train_images])  # 将每个图像调整为20x20
train_images = train_images.reshape((60000, 20, 20, 1))  # 为每个图像增加通道维度
train_images = train_images.astype('float32') / 255  # 将像素值归一化到0-1之间

test_images = test_images.reshape((10000, 28, 28))  # 同样处理测试数据
test_images = np.asarray([np.asarray(Image.fromarray(image).resize((20, 20))) for image in test_images])
test_images = test_images.reshape((10000, 20, 20, 1))
test_images = test_images.astype('float32') / 255

# 将标签转换为独热编码格式
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 设置模型参数
input_shape = (20, 20, 1)  # 输入图像的形状
num_classes = 10  # 分类数目

# 创建CNN模型
model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_constraint=weight_constraint))  # 卷积层
model.add(MaxPooling2D((4, 4)))  # 最大池化层
model.add(Flatten())  # 展平层
model.add(Dense(num_classes, activation='softmax', input_shape=(400,), kernel_constraint=weight_constraint))  # 全连接层，输出类别数为10
model.summary()  # 输出模型结构概览

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
batch_accuracy_callback = BatchAccuracyCallback()  # 初始化回调函数
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), callbacks=[batch_accuracy_callback])

# 获取全连接层的权重并绘制直方图
dense_layer_weights = model.layers[-1].get_weights()[0]  # 获取最后一层（全连接层）的权重
plt.figure(1)
plt.hist(dense_layer_weights.flatten(), bins=50, color='blue')  # 绘制权重的直方图
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Histogram of Dense Layer Weights')
plt.show()

# 将权重保存为CSV文件
np.savetxt('dense_layer_weights.csv', dense_layer_weights, delimiter=',')

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)  # 在测试集上评估模型
print('Test accuracy:', test_acc)

# 绘制训练和验证准确率曲线
plt.figure(2)
plt.plot(history.history['accuracy'], label='Training accuracy')  # 绘制训练准确率曲线
plt.plot(history.history['val_accuracy'], label='Validation accuracy')  # 绘制验证准确率曲线
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 绘制每个批次的训练准确率曲线
plt.figure(3)
plt.plot(batch_accuracy_callback.batch_accuracies, label='Training accuracy per batch')  # 绘制每个批次的训练准确率
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy per Batch')
plt.legend()
plt.show()

# 选择一张图片并绘制热图
image_index = 1  # 选择索引为1的图片
heatmap = train_images[image_index]  # 获取该图片
plt.figure()
plt.imshow(heatmap, cmap='hot')  # 绘制热图
plt.colorbar()
plt.show()

# 对模型应用4位量化
quantize_model = tfmot.quantization.keras.quantize_annotate_model(model)  # 注解模型，准备进行量化
quantize_model = tfmot.quantization.keras.quantize_apply(quantize_model)  # 应用量化

# 编译量化后的模型
quantize_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练量化后的模型
history_quantized = quantize_model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), callbacks=[batch_accuracy_callback])

# 评估量化后的模型
test_loss_quantized, test_acc_quantized = quantize_model.evaluate(test_images, test_labels)  # 在测试集上评估量化模型
print('Quantized test accuracy:', test_acc_quantized)
