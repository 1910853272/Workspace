import cv2
import matplotlib.pyplot as plt

# 1. 读取图像
img = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)

# 2. 高斯模糊
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# 3. Canny边缘检测
edges = cv2.Canny(blurred, threshold1=5, threshold2=20)


cv2.imwrite('2_c.png', edges)
