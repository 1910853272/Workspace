import numpy as np

# 创建字母U矩阵
matrix = np.array([[1., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 1.],
                   [1., 1., 1., 1., 1.]])

# 保存为 .npy 文件
np.save('U.npy', matrix)

# 加载并打印矩阵验证保存成功
loaded_matrix = np.load('U.npy')
print(loaded_matrix)
