import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image = cv2.imread('image1.png')  # 替换为图像的路径

# 2. 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 进行Gamma变换
gamma = 0.5  # 设置gamma值，可以调节增强效果
c = 255 / (np.max(gray_image) ** gamma)  # 计算常数c，确保增强后的图像不超过255

# Gamma变换公式 S = c * I^gamma
gamma_image = c * (gray_image ** gamma)

# 4. 将结果转换回8位整数类型
gamma_image = np.array(gamma_image, dtype=np.uint8)

# 5. 显示结果
plt.imshow(gamma_image, cmap='gray')
plt.axis('off')  # 关闭坐标轴显示

# 6. 保存图像
plt.savefig('image1_Gamma.png', bbox_inches='tight', pad_inches=0)
plt.show()

