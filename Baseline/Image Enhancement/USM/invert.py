import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('c.jpg', cv2.IMREAD_GRAYSCALE)

# 非线性灰度变换 - 反转（负片）
inverted_img = 255 - img

cv2.imwrite('inverted_c.png', inverted_img)


