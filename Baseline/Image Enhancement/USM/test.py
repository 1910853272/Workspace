import cv2 as cv
import numpy as np

# 读取灰度图像（二值图像）
image = cv.imread("Binary2.png", cv.IMREAD_GRAYSCALE)

# 1. 二值化（确保图像只有黑白两色）
_, binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# 2. 查找所有外部轮廓
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 3. 新建黑色背景图（全零）
cleaned = np.zeros_like(binary)

# 4. 遍历轮廓，只绘制大于 min_area 的
min_area = 0  # 可根据实际图像调整
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > min_area:
        cv.drawContours(cleaned, [cnt], -1, 255, thickness=cv.FILLED)

# 显示和保存结果
cv.imshow("Cleaned Image", cleaned)
cv.imwrite("cleaned_output.png", cleaned)  # 可选：保存图像
cv.waitKey(0)
cv.destroyAllWindows()
