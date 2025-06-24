import numpy as np

# ── 原始矩阵 human1 ──
human1 = np.array([
    [0, 0, 0, 9, 0],
    [0, 0, 7, 8, 7],
    [0, 0, 0, 8, 0],
    [0, 0, 8, 0, 8],
    [0, 0, 6, 0, 6]
], dtype=float)

tau, beta, T = 1.704, 1.0, 1      # 衰减参数
decay_factor = np.exp(-(T/tau)**beta)

def shift_left(matrix, n):
    """将矩阵左移n列。"""
    res = np.zeros_like(matrix)
    if n > 0 and n < matrix.shape[1]:
        res[:, :-n] = matrix[:, n:]
    return res

# 生成前10帧（索引0-9）
frames = [human1.copy()]
for i in range(1, 10):
    prev = frames[-1]
    # 对前一帧进行指数衰减
    decayed = np.where(prev > 0, prev * decay_factor, 0.0)
    # 将原始矩阵向左平移 i 列
    shifted = shift_left(human1, i)
    # 叠加得到新帧
    new_frame = decayed + shifted
    frames.append(new_frame)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D

# 输入为三帧拼接后的5×5图像，通道数为3
input_img = Input(shape=(5, 5, 3))

# 编码器部分
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)  # padding='same' 保持尺寸
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# 解码器部分
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 输出为5x5图像

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 构造训练数据 X_train, Y_train
X_train, Y_train = [], []
for i in range(len(frames) - 3):
    # 连续3帧堆叠为输入
    seq3 = np.stack((frames[i], frames[i+1], frames[i+2]), axis=-1)
    X_train.append(seq3)
    # 第4帧作为标签
    Y_train.append(frames[i+3][..., np.newaxis])

# 转为数组并归一化（假设像素值范围在0-10）
X_train = np.array(X_train) / 10.0
Y_train = np.array(Y_train) / 10.0

# 训练模型
autoencoder.fit(X_train, Y_train, epochs=20, batch_size=1)

# 将前3帧构造为模型输入并归一化
input_seq = np.stack((frames[0], frames[1], frames[2]), axis=-1) / 10.0
predicted_frames = []
for _ in range(8):
    # 增加批次维度并预测下一帧
    pred = autoencoder.predict(input_seq[np.newaxis, ...])
    pred = np.squeeze(pred)  # 去掉批次维度
    # 通过expand_dims添加通道维度，使其变成 (5, 5, 1)
    pred = np.expand_dims(pred, axis=-1)  # 确保维度是 (5, 5, 1)
    # 存储反归一化的预测结果
    predicted_frames.append(pred * 10.0)
    # 更新输入序列：抛弃最旧一帧，加入新预测帧
    input_seq = np.concatenate((input_seq[..., 1:], pred), axis=-1)

# 可视化预测结果
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = ['#FFFFFF', '#ABCBE4', '#6BA3CF', '#2B7BBA']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors)

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()
for i, frame in enumerate(predicted_frames):
    ax = axes[i]
    ax.imshow(frame, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
    ax.set_title(f'Predict {i+4}')
    ax.set_xticks([]); ax.set_yticks([])

# 关闭空白子图
for j in range(len(predicted_frames), 8):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
