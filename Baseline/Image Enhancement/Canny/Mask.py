import cv2 as cv
import numpy as np

# 方法1：直接灰度 + Otsu二值化
def method_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# 方法2：高斯模糊 + 灰度 + Otsu二值化
def method_2(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# 方法3：均值漂移 + 灰度 + Otsu二值化
def method_3(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

# 读取原图像
src = cv.imread("2.png")

# 选择一种方法处理（可切换为 method_1 / method_2 / method_3）
ret = method_1(src)

# 取反二值图，黑变白，白变黑.只保留黑色区域对应的原图像素
ret_inv = cv.bitwise_not(ret)

# 使用 binary 作为掩膜，只保留白色区域对应的原图像素
masked = cv.bitwise_and(src, src, mask=ret_inv)

# 显示最终结果
cv.imwrite("2Binary.png", ret)      # 保存二值图
cv.imwrite("2Masked.png", masked)   # 保存掩膜图

