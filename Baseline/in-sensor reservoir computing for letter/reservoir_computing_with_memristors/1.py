import numpy as np
import pandas as pd
import cv2

# 从Excel读取输入的字母数据，并进行转置以便于后续处理
letter_df = pd.read_excel('letter.xlsx', header=None)
letter = letter_df.values.T  # 转置数据
target = np.eye(20)  # 定义目标向量（单位矩阵）

larger = 10  # 数据集扩展因子
row, col = letter.shape  # 获取输入数据的行列数

# 通过添加椒盐噪声来创建更大的数据集
letters_expanded = [letter]
for i in range(larger - 1):
    noisy_letter = letter.copy()
    for j in range(noisy_letter.shape[1]):
        noisy_letter[:, j] = cv2.randn(noisy_letter[:, j], 0, 255)  # 添加椒盐噪声
    letters_expanded.append(noisy_letter)
letter = np.concatenate(letters_expanded, axis=1)  # 将噪声数据添加到字母数据中

# 扩展目标矩阵以匹配增大的数据集
target_expanded = [target]
for i in range(larger - 1):
    target_expanded.append(target)
target = np.concatenate(target_expanded, axis=1)

# 定义每个输入的状态向量（根据问题定义）
state = [0.0, 8.314, 7.603, 16.859, 6.917, 16.816, 14.823, 24.789,
         7.115, 16.177, 14.101, 25.118, 15.333, 23.401, 22.758, 32.286]
inputs = np.zeros((5, 20 * larger))  # 初始化输入矩阵

# 将字母数据映射为状态向量中的值，构建新的输入矩阵
for i in range(20 * larger):
    for j in range(5):
        order = (letter[i, 4 * j - 3] * 8 + letter[i, 4 * j - 2] * 4 +
                 letter[i, 4 * j - 1] * 2 + letter[i, 4 * j])
        inputs[j, i] = state[int(order)]  # 将状态值赋给输入矩阵

# 打印输出
print("输入矩阵：")
print(inputs)
print("目标矩阵：")
print(target)
