import cv2 as cv
import numpy as np
#非锐化掩膜USM(Unsharpen Mask)
#先对原图高斯模糊，用原图减去系数x高斯模糊的图像
#再把值Scale到0~255的RGB像素范围
#优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
#（原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
src = cv.imread("8.png")

# sigma = 5、15、25
# 使用 高斯模糊处理原图，(0,0) 表示由 sigma=5 决定核大小，模糊结果保存在 blur_img。
blur_img = cv.GaussianBlur(src, (0, 0), 5)
# 使用加权叠加方式得到锐化图像：usm=1.5⋅src−0.5⋅blur_img 本质是：原图减去一定权重的模糊图，强化边缘和细节。
#cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
usm = cv.addWeighted(src, 1.5, blur_img, -0.6, 0)

# 保存结果图像
cv.imwrite("8USM.png", usm)