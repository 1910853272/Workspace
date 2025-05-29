import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

# 使用拉普拉斯算子进行边缘检测
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# 取绝对值并转换为uint8
laplacian_abs = cv2.convertScaleAbs(laplacian)

# 边缘增强：原图像加上拉普拉斯算子结果
enhanced = cv2.addWeighted(img, 1, laplacian_abs, 1, 0)

cv2.imwrite('original.png', img)
cv2.imwrite('laplacian.png', laplacian_abs)
cv2.imwrite('enhanced.png', enhanced)

